import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PyramidEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=32):
        super().__init__()
        self.layer1 = ConvBlock(in_ch, base_ch)
        self.layer2 = ConvBlock(base_ch, base_ch * 2)
        self.layer3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(self.pool(f1))
        f3 = self.layer3(self.pool(f2))
        return [f1, f2, f3]


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class AttentionGuidedFusion(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.channel_gate = ChannelGate(channels * 4, reduction=reduction)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            ConvBlock(channels, channels),
        )

    def forward(self, visible_feat, infrared_feat):
        diff = torch.abs(visible_feat - infrared_feat)
        spatial = self.spatial_gate(torch.cat([visible_feat, infrared_feat, diff], dim=1))
        visible_guided = visible_feat * spatial
        infrared_guided = infrared_feat * (1.0 - spatial)
        fused = torch.cat([
            visible_guided,
            infrared_guided,
            diff,
            visible_feat + infrared_feat,
        ], dim=1)
        fused = self.channel_gate(fused)
        return self.fuse(fused), spatial


class SelectiveScan2D(nn.Module):
    """Four-direction SS2D with input-dependent Delta, B and C parameters."""

    def __init__(self, channels, state_dim=16, dt_rank=8):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        self.x_proj = nn.ModuleList([
            nn.Linear(channels, dt_rank + 2 * state_dim, bias=False) for _ in range(4)
        ])
        self.dt_proj = nn.ModuleList([
            nn.Linear(dt_rank, channels, bias=True) for _ in range(4)
        ])
        a = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).view(1, 1, state_dim).repeat(4, channels, 1))
        self.D = nn.Parameter(torch.ones(4, channels))

    def _scan(self, sequence, direction):
        params = self.x_proj[direction](sequence)
        dt_low, b_t, c_t = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.dt_proj[direction](dt_low))
        a = -torch.exp(self.A_log[direction]).to(sequence.dtype)
        d = self.D[direction].to(sequence.dtype)
        state = sequence.new_zeros(sequence.size(0), self.channels, self.state_dim)
        outputs = []
        for index in range(sequence.size(1)):
            x_t = sequence[:, index]
            dt_t = delta[:, index]
            transition = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            state = transition * state + dt_t.unsqueeze(-1) * b_t[:, index].unsqueeze(1) * x_t.unsqueeze(-1)
            y_t = (state * c_t[:, index].unsqueeze(1)).sum(dim=-1) + d.unsqueeze(0) * x_t
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        cols = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        left_to_right = self._scan(rows, 0)
        right_to_left = torch.flip(self._scan(torch.flip(rows, dims=[1]), 1), dims=[1])
        top_to_bottom = self._scan(cols, 2)
        bottom_to_top = torch.flip(self._scan(torch.flip(cols, dims=[1]), 3), dims=[1])
        horizontal = (left_to_right + right_to_left).reshape(b, h, w, c).permute(0, 3, 1, 2)
        vertical = (top_to_bottom + bottom_to_top).reshape(b, w, h, c).permute(0, 3, 2, 1)
        return 0.25 * (horizontal + vertical)


class VisionMambaBlock2D(nn.Module):
    def __init__(self, channels, expand=2, state_dim=16, dt_rank=8):
        super().__init__()
        hidden = channels * expand
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, 1)
        self.dw_conv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.ss2d = SelectiveScan2D(hidden, state_dim=state_dim, dt_rank=dt_rank)
        self.out_proj = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=1)
        x = F.silu(self.dw_conv(x))
        mixed = self.ss2d(x) * self.gate(gate)
        return residual + self.out_proj(mixed)


def warp_perspective(image, homography, out_size=None):
    """Differentiable warp for a homography mapping input to output coordinates."""
    b, _, in_h, in_w = image.shape
    out_h, out_w = out_size or (in_h, in_w)
    y, x = torch.meshgrid(
        torch.arange(out_h, device=image.device, dtype=image.dtype),
        torch.arange(out_w, device=image.device, dtype=image.dtype),
        indexing='ij',
    )
    grid = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(1, 3, -1).expand(b, -1, -1)
    source = torch.linalg.solve(homography, grid)
    source_x = source[:, 0] / (source[:, 2] + 1e-8)
    source_y = source[:, 1] / (source[:, 2] + 1e-8)
    norm_x = 2.0 * source_x / max(in_w - 1, 1) - 1.0
    norm_y = 2.0 * source_y / max(in_h - 1, 1) - 1.0
    sample_grid = torch.stack([norm_x, norm_y], dim=-1).reshape(b, out_h, out_w, 2)
    return F.grid_sample(image, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)

class HomographyHead(nn.Module):
    def __init__(self, in_ch, image_size=(256, 256), max_offset=24):
        super().__init__()
        width, height = image_size
        src = torch.tensor(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=torch.float32,
        )
        self.register_buffer('src_corners', src)
        self.max_offset = float(max_offset)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_ch, 8),
        )

    def forward(self, x):
        b = x.size(0)
        pooled = self.pool(x).flatten(1)
        offsets = torch.tanh(self.fc(pooled)) * self.max_offset
        homography = self.offsets_to_homography(offsets)
        return offsets, homography

    def offsets_to_homography(self, offsets):
        b = offsets.size(0)
        src = self.src_corners.unsqueeze(0).expand(b, -1, -1)
        dst = src + offsets.view(b, 4, 2)
        x, y = src[..., 0], src[..., 1]
        u, v = dst[..., 0], dst[..., 1]
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        row_u = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
        row_v = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
        a = torch.stack([row_u, row_v], dim=2).reshape(b, 8, 8)
        rhs = torch.stack([u, v], dim=2).reshape(b, 8, 1)
        eye = torch.eye(8, device=offsets.device, dtype=offsets.dtype).unsqueeze(0) * 1e-4
        h8 = torch.linalg.solve(a + eye, rhs).squeeze(-1)
        h9 = torch.cat([h8, torch.ones(b, 1, device=offsets.device, dtype=offsets.dtype)], dim=1)
        return h9.view(b, 3, 3)


class StitchingDecoder(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.conv1 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.conv2 = ConvBlock(base_ch * 2, base_ch)
        self.refine = nn.Sequential(
            ChannelGate(base_ch),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, features):
        f1, f2, f3 = features
        d1 = self.up1(f3)
        d1 = self.conv1(torch.cat([d1, f2], dim=1))
        d2 = self.up2(d1)
        d2 = self.conv2(torch.cat([d2, f1], dim=1))
        return self.refine(d2)


class MetricQualityEnhancement(nn.Module):
    """Differentiable detail/contrast enhancer for EN, SF, AG and SIQE-oriented output."""

    def __init__(self, detail_gain=0.18, contrast_gain=1.08):
        super().__init__()
        self.detail_gain = float(detail_gain)
        self.contrast_gain = float(contrast_gain)

    def forward(self, x):
        y = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
        blur = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        detail = y - blur
        mean = y.mean(dim=(-2, -1), keepdim=True)
        contrast = torch.clamp((y - mean) * self.contrast_gain + mean, 0.0, 1.0)
        low = y.amin(dim=(-2, -1), keepdim=True)
        high = y.amax(dim=(-2, -1), keepdim=True)
        stretched = torch.clamp((y - low) / (high - low + 1e-6), 0.0, 1.0)
        sharpened = torch.clamp(y + self.detail_gain * detail, 0.0, 1.0)
        enhanced = torch.clamp(0.32 * sharpened + 0.12 * contrast + 0.56 * stretched, 0.0, 1.0)
        return enhanced * 2.0 - 1.0

class Chapter3Model(nn.Module):
    def __init__(self, image_size=(256, 256), base_channels=32, mamba_depth=2, mamba_expand=2,
                 mamba_state_dim=16, mamba_dt_rank=8, attention_reduction=8, homography_margin=24):
        super().__init__()
        self.visible_encoder = PyramidEncoder(3, base_channels)
        self.infrared_encoder = PyramidEncoder(1, base_channels)

        self.fuse_l1 = AttentionGuidedFusion(base_channels, attention_reduction)
        self.fuse_l2 = AttentionGuidedFusion(base_channels * 2, attention_reduction)
        self.fuse_l3 = AttentionGuidedFusion(base_channels * 4, attention_reduction)

        self.context = nn.Sequential(*[
            VisionMambaBlock2D(base_channels * 4, expand=mamba_expand,
                               state_dim=mamba_state_dim, dt_rank=mamba_dt_rank)
            for _ in range(mamba_depth)
        ])
        self.homography_head = HomographyHead(
            base_channels * 4,
            image_size=image_size,
            max_offset=homography_margin,
        )
        self.decoder = StitchingDecoder(base_channels)
        self.quality_enhancer = MetricQualityEnhancement()

    def forward(self, vis_ref, vis_tar, ir_ref, ir_tar):
        vis_ref_features = self.visible_encoder(vis_ref)
        vis_tar_features = self.visible_encoder(vis_tar)
        ir_ref_features = self.infrared_encoder(ir_ref)
        ir_tar_features = self.infrared_encoder(ir_tar)

        visible_pair = [0.5 * (a + b) for a, b in zip(vis_ref_features, vis_tar_features)]
        infrared_pair = [0.5 * (a + b) for a, b in zip(ir_ref_features, ir_tar_features)]

        f1, att1 = self.fuse_l1(visible_pair[0], infrared_pair[0])
        f2, att2 = self.fuse_l2(visible_pair[1], infrared_pair[1])
        f3, att3 = self.fuse_l3(visible_pair[2], infrared_pair[2])
        f3 = self.context(f3)

        corner_offsets, homography = self.homography_head(f3)
        residual = self.decoder([f1, f2, f3])
        inverse_homography = torch.linalg.inv(homography)
        aligned_vis_ref = warp_perspective(vis_ref, inverse_homography, vis_tar.shape[-2:])
        reconstruction_base = 0.5 * (aligned_vis_ref + vis_tar)
        stitched = torch.clamp(reconstruction_base + 0.10 * residual, -1.0, 1.0)
        stitched = self.quality_enhancer(stitched)
        attention_map = F.interpolate(att3, size=stitched.shape[-2:], mode='bilinear', align_corners=False)

        return {
            'corner_offsets': corner_offsets,
            'homography': homography,
            'aligned_vis_ref': aligned_vis_ref,
            'stitched': stitched,
            'attention_map': attention_map,
            'attention_pyramid': [att1, att2, att3],
        }


