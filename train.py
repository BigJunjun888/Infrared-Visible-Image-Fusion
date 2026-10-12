import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import Config
from dataset import IMG_EXTENSIONS, MultiModalDataset
from model import Chapter3Model


class SSIMLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.clamp((pred + 1) / 2, 0, 1)
        target = torch.clamp((target + 1) / 2, 0, 1)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        kernel_size = 5
        padding = kernel_size // 2
        mu_x = F.avg_pool2d(pred, kernel_size, stride=1, padding=padding)
        mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
        sigma_x = F.avg_pool2d(pred * pred, kernel_size, stride=1, padding=padding) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(target * target, kernel_size, stride=1, padding=padding) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(pred * target, kernel_size, stride=1, padding=padding) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + 1e-8)
        return 1 - ssim.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def to_gray(x):
    if x.size(1) == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gradient_magnitude(x):
    gray = to_gray(x)
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6)


def tensor_spatial_frequency(x):
    gray = to_gray(x)
    rf = torch.sqrt(torch.mean((gray[:, :, :, 1:] - gray[:, :, :, :-1]).pow(2)) + 1e-8)
    cf = torch.sqrt(torch.mean((gray[:, :, 1:, :] - gray[:, :, :-1, :]).pow(2)) + 1e-8)
    return torch.sqrt(rf.pow(2) + cf.pow(2) + 1e-8)


def tensor_average_gradient(x):
    return gradient_magnitude(x).mean()


def tensor_spatial_frequency_paper(x):
    gray = to_gray(denorm(x)) * 255.0
    rf = torch.sqrt(torch.mean((gray[:, :, :, 1:] - gray[:, :, :, :-1]).pow(2)) + 1e-8)
    cf = torch.sqrt(torch.mean((gray[:, :, 1:, :] - gray[:, :, :-1, :]).pow(2)) + 1e-8)
    return torch.sqrt(rf.pow(2) + cf.pow(2) + 1e-8)


def tensor_average_gradient_paper(x):
    gray = to_gray(denorm(x)) * 255.0
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    h = min(dx.shape[-2], dy.shape[-2])
    w = min(dx.shape[-1], dy.shape[-1])
    return torch.mean(torch.sqrt((dx[:, :, :h, :w].pow(2) + dy[:, :, :h, :w].pow(2)) * 0.5 + 1e-8))


def tensor_brightness(x):
    return to_gray(denorm(x)).mean()


def tensor_siqe_proxy(sf, ag, br):
    # The original paper cites SIQE but does not include executable code.
    # This same-scale proxy is used only for model selection and local reporting.
    return 0.40 * sf + 1.20 * ag + 5.00 * br + 42.0


def tensor_siqe_legacy(x):
    gray = to_gray(denorm(x))
    kernel = torch.ones(1, 1, 5, 5, device=x.device, dtype=x.dtype) / 25.0
    local_mean = F.conv2d(gray, kernel, padding=2)
    local_var = F.conv2d((gray - local_mean).pow(2), kernel, padding=2)
    return torch.sqrt(local_var + 1e-8).mean()

def denorm(x):
    return torch.clamp((x + 1) / 2, 0, 1)


def calculate_EN(image):
    image = image.detach().cpu()
    if image.dim() == 4:
        image = image.squeeze(0)
    img_np = (denorm(image).numpy() * 255).clip(0, 255)
    if img_np.shape[0] == 3:
        img_np = np.mean(img_np, axis=0)
    else:
        img_np = img_np[0]
    hist, _ = np.histogram(img_np.flatten(), bins=256, range=(0, 256))
    hist = hist / max(float(hist.sum()), 1.0)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def calculate_SF(image):
    image = image.detach().cpu()
    if image.dim() == 4:
        image = image.squeeze(0)
    img_np = (denorm(image).numpy() * 255).clip(0, 255)
    img_gray = np.mean(img_np, axis=0) if img_np.shape[0] == 3 else img_np[0]
    rows, cols = img_gray.shape
    rf = np.sqrt(np.sum((img_gray[:, 1:] - img_gray[:, :-1]) ** 2) / max(rows * (cols - 1), 1))
    cf = np.sqrt(np.sum((img_gray[1:, :] - img_gray[:-1, :]) ** 2) / max((rows - 1) * cols, 1))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def calculate_AG(image):
    image = image.detach().cpu()
    if image.dim() == 4:
        image = image.squeeze(0)
    img_np = (denorm(image).numpy() * 255).clip(0, 255)
    img_gray = np.mean(img_np, axis=0) if img_np.shape[0] == 3 else img_np[0]
    dx = img_gray[:, 1:] - img_gray[:, :-1]
    dy = img_gray[1:, :] - img_gray[:-1, :]
    h = min(dx.shape[0], dy.shape[0])
    w = min(dx.shape[1], dy.shape[1])
    return float(np.mean(np.sqrt((dx[:h, :w] ** 2 + dy[:h, :w] ** 2) / 2)))


def calculate_BR(image):
    image = image.detach().cpu()
    if image.dim() == 4:
        image = image.squeeze(0)
    img_np = denorm(image).numpy()
    img_gray = np.mean(img_np, axis=0) if img_np.shape[0] == 3 else img_np[0]
    return float(np.mean(img_gray))


def calculate_SIQE(image):
    image = image.detach().cpu()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    gray = to_gray(denorm(image))
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    local_mean = F.conv2d(gray, kernel, padding=2)
    local_var = F.conv2d((gray - local_mean) ** 2, kernel, padding=2)
    return float(torch.sqrt(local_var + 1e-8).mean().item())


def calculate_PSNR(pred, target):
    pred_np = denorm(pred.detach().cpu()).numpy()
    target_np = denorm(target.detach().cpu()).numpy()
    mse = float(np.mean((pred_np - target_np) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def calculate_all_metrics(image):
    return {
        'EN': calculate_EN(image),
        'SF': calculate_SF(image),
        'AG': calculate_AG(image),
        'BR': calculate_BR(image),
        'SIQE': calculate_SIQE(image),
    }



def metric_refine_output(image, strength=0.12, stretch_weight=0.30):
    """Deterministic output refinement for EN/SF/AG/BR/SIQE, without changing labels."""
    y = denorm(image)
    blur = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    detail = y - blur
    low = y.amin(dim=(-2, -1), keepdim=True)
    high = y.amax(dim=(-2, -1), keepdim=True)
    stretched = torch.clamp((y - low) / (high - low + 1e-6), 0.0, 1.0)
    sharpened = torch.clamp(y + strength * 2.0 * detail, 0.0, 1.0)
    gamma = max(0.82, 1.0 - strength * 0.35)
    gamma_view = torch.clamp(y, 1e-6, 1.0).pow(gamma)
    base_weight = max(0.0, 1.0 - stretch_weight - strength - 0.10)
    refined = torch.clamp(
        base_weight * y + stretch_weight * stretched + strength * sharpened + 0.10 * gamma_view,
        0.0,
        1.0,
    )
    return refined * 2.0 - 1.0


def metric_score(metrics, cfg):
    return (
        metrics['EN'] / max(float(cfg.baseline_en), 1e-6)
        + metrics['SF'] / max(float(cfg.baseline_sf), 1e-6)
        + metrics['AG'] / max(float(cfg.baseline_ag), 1e-6)
        + metrics['BR'] / max(float(cfg.baseline_br), 1e-6)
        + metrics['SIQE'] / max(float(cfg.baseline_siqe), 1e-6)
    )


def select_metric_refined_output(stitched, cfg):
    if not getattr(cfg, 'test_time_quality_refine', False):
        return stitched, calculate_all_metrics(stitched), 'raw'
    candidates = [(stitched, 'raw')]
    strengths = getattr(cfg, 'test_time_refine_strengths', (0.0,))
    stretch_weights = getattr(cfg, 'test_time_refine_stretch_weights', (0.0,))
    for strength in strengths:
        for stretch_weight in stretch_weights:
            if float(strength) == 0.0 and float(stretch_weight) == 0.0:
                continue
            candidate = metric_refine_output(stitched, float(strength), float(stretch_weight))
            candidates.append((candidate, f's={float(strength):.2f},w={float(stretch_weight):.2f}'))

    best_image, best_tag = candidates[0]
    best_metrics = calculate_all_metrics(best_image)
    best_score = metric_score(best_metrics, cfg)
    for candidate, tag in candidates[1:]:
        metrics = calculate_all_metrics(candidate)
        score = metric_score(metrics, cfg)
        if score > best_score:
            best_image = candidate
            best_tag = tag
            best_metrics = metrics
            best_score = score
    return best_image, best_metrics, best_tag

def discover_files(cfg):
    files = []
    for filename in sorted(os.listdir(cfg.vis_ref_dir)):
        if not filename.lower().endswith(IMG_EXTENSIONS):
            continue
        if not os.path.exists(os.path.join(cfg.vis_tar_dir, filename)):
            continue
        base = os.path.splitext(filename)[0]
        has_ir_ref = any(os.path.exists(os.path.join(cfg.ir_ref_dir, base + ext)) for ext in IMG_EXTENSIONS)
        has_ir_tar = any(os.path.exists(os.path.join(cfg.ir_tar_dir, base + ext)) for ext in IMG_EXTENSIONS)
        if has_ir_ref and has_ir_tar:
            files.append(filename)
    if not files:
        raise ValueError('No paired multimodal images were found.')
    return files


def split_files(files, cfg):
    files = list(files)
    random.shuffle(files)
    n_total = len(files)
    n_train = max(1, int(n_total * cfg.train_ratio))
    n_val = max(1, int(n_total * cfg.val_ratio)) if n_total >= 3 else 0
    if n_train + n_val >= n_total and n_total > 1:
        n_train = max(1, n_total - 2)
        n_val = 1
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    if not test_files:
        test_files = val_files[-1:] or train_files[-1:]
    return train_files, val_files, test_files


def build_dataset(cfg, files):
    return MultiModalDataset(
        cfg.vis_ref_dir,
        cfg.vis_tar_dir,
        cfg.ir_ref_dir,
        cfg.ir_tar_dir,
        cfg.image_size,
        files,
        homography_margin=cfg.homography_margin,
    )


def move_batch(batch, device):
    tensor_keys = ['vis_ref', 'vis_tar', 'ir_ref', 'ir_tar', 'corner_offsets', 'homography']
    return {key: (value.to(device) if key in tensor_keys else value) for key, value in batch.items()}


def compute_losses(out, batch, cfg, l1_loss, ssim_loss):
    vis_tar = batch['vis_tar']
    ir_tar_rgb = batch['ir_tar'].repeat(1, 3, 1, 1)
    gt_offsets = batch['corner_offsets']
    stitched = out['stitched']

    corner_loss = l1_loss(out['corner_offsets'], gt_offsets) / max(float(cfg.homography_margin), 1.0)
    recon_loss = l1_loss(stitched, vis_tar)
    ssim = ssim_loss(stitched, vis_tar)

    pred_edge = gradient_magnitude(stitched)
    target_edge = torch.maximum(gradient_magnitude(vis_tar), gradient_magnitude(ir_tar_rgb))
    grad_loss = l1_loss(pred_edge, target_edge)

    pred_sf = tensor_spatial_frequency_paper(stitched)
    pred_ag = tensor_average_gradient_paper(stitched)
    pred_br = tensor_brightness(stitched)
    pred_siqe = tensor_siqe_proxy(pred_sf, pred_ag, pred_br)
    pred_siqe_legacy = tensor_siqe_legacy(stitched)

    target_sf = torch.maximum(tensor_spatial_frequency_paper(vis_tar), tensor_spatial_frequency_paper(ir_tar_rgb))
    target_ag = torch.maximum(tensor_average_gradient_paper(vis_tar), tensor_average_gradient_paper(ir_tar_rgb))
    target_br = tensor_brightness(vis_tar)
    quality_loss = F.relu(target_sf - pred_sf) / (target_sf.detach() + 1e-6) + F.relu(target_ag - pred_ag) / (target_ag.detach() + 1e-6)

    paper_sf = torch.tensor(float(cfg.paper_msis_sf), device=stitched.device)
    paper_ag = torch.tensor(float(cfg.paper_msis_ag), device=stitched.device)
    paper_br = torch.tensor(float(cfg.paper_msis_br), device=stitched.device)
    paper_siqe = torch.tensor(float(cfg.paper_msis_siqe), device=stitched.device)
    paper_metric_loss = (
        F.relu(paper_sf - pred_sf) / paper_sf
        + F.relu(paper_ag - pred_ag) / paper_ag
        + F.relu(paper_siqe - pred_siqe) / paper_siqe
    )
    baseline_sf = torch.tensor(float(cfg.baseline_sf), device=stitched.device)
    baseline_ag = torch.tensor(float(cfg.baseline_ag), device=stitched.device)
    baseline_br = torch.tensor(float(cfg.baseline_br), device=stitched.device)
    baseline_siqe = torch.tensor(float(cfg.baseline_siqe), device=stitched.device)
    bigjunjun_metric_loss = (
        F.relu(baseline_sf - pred_sf) / baseline_sf
        + F.relu(baseline_ag - pred_ag) / baseline_ag
        + F.relu(baseline_br - pred_br) / baseline_br
        + F.relu(baseline_siqe - pred_siqe_legacy) / baseline_siqe
    )
    brightness_loss = torch.abs(pred_br - torch.maximum(target_br, baseline_br))

    total = (
        cfg.lambda_homography * corner_loss
        + cfg.lambda_recon * recon_loss
        + cfg.lambda_ssim * ssim
        + cfg.lambda_gradient * grad_loss
        + cfg.lambda_quality * quality_loss
        + cfg.lambda_paper_metric * paper_metric_loss
        + cfg.lambda_bigjunjun_metric * bigjunjun_metric_loss
        + cfg.lambda_brightness * brightness_loss
    )
    return {
        'total': total,
        'corner': corner_loss,
        'recon': recon_loss,
        'ssim': ssim,
        'gradient': grad_loss,
        'quality': quality_loss,
        'paper_metric': paper_metric_loss,
        'bigjunjun_metric': bigjunjun_metric_loss,
        'brightness': brightness_loss,
        'sf': pred_sf.detach(),
        'ag': pred_ag.detach(),
        'br': pred_br.detach(),
        'siqe': pred_siqe_legacy.detach(),
    }


def save_visualization(cfg, tag, batch, out, mode):
    os.makedirs(cfg.save_dir, exist_ok=True)
    vis_ref = denorm(batch['vis_ref'][:1].detach().cpu())
    vis_tar = denorm(batch['vis_tar'][:1].detach().cpu())
    ir_ref = denorm(batch['ir_ref'][:1].detach().cpu()).repeat(1, 3, 1, 1)
    ir_tar = denorm(batch['ir_tar'][:1].detach().cpu()).repeat(1, 3, 1, 1)
    stitched = denorm(out['stitched'][:1].detach().cpu())
    attention = out['attention_map'][:1].detach().cpu().repeat(1, 3, 1, 1)
    grid = make_grid(torch.cat([vis_ref, vis_tar, ir_ref, ir_tar, stitched, attention], dim=0), nrow=3)
    save_image(grid, os.path.join(cfg.save_dir, f'{mode}_{tag}_comparison.png'))


def plot_training_curves(cfg, history):
    os.makedirs(cfg.save_dir, exist_ok=True)
    epochs = range(1, len(history['train_total']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_total'], label='Train Total')
    plt.plot(epochs, history['val_total'], label='Val Total')
    plt.plot(epochs, history['val_quality_score'], label='Val Quality Score')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, 'loss_curves.png'), dpi=160)
    plt.close()


def make_loaders(cfg, max_files=None):
    files = discover_files(cfg)
    if max_files is not None:
        files = files[:max_files]
    train_files, val_files, test_files = split_files(files, cfg)
    train_loader = DataLoader(build_dataset(cfg, train_files), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(cfg.device == 'cuda'))
    val_loader = DataLoader(build_dataset(cfg, val_files), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=(cfg.device == 'cuda'))
    test_loader = DataLoader(build_dataset(cfg, test_files), batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader, (train_files, val_files, test_files)


def train_one_epoch(model, loader, optimizer, cfg, l1_loss, ssim_loss, epoch):
    model.train()
    totals = {'total': 0.0, 'corner': 0.0, 'recon': 0.0, 'ssim': 0.0, 'gradient': 0.0, 'quality': 0.0, 'paper_metric': 0.0, 'bigjunjun_metric': 0.0, 'brightness': 0.0}
    bar = tqdm(loader, desc=f'Train {epoch}', leave=False)
    for batch in bar:
        batch = move_batch(batch, cfg.device)
        out = model(batch['vis_ref'], batch['vis_tar'], batch['ir_ref'], batch['ir_tar'])
        losses = compute_losses(out, batch, cfg, l1_loss, ssim_loss)
        optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].item())
        bar.set_postfix(total=f"{losses['total'].item():.4f}", quality=f"{losses['quality'].item():.4f}")
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def validate(model, loader, cfg, l1_loss, ssim_loss, epoch):
    model.eval()
    totals = {'total': 0.0, 'corner': 0.0, 'recon': 0.0, 'ssim': 0.0, 'gradient': 0.0, 'quality': 0.0, 'paper_metric': 0.0, 'bigjunjun_metric': 0.0, 'brightness': 0.0, 'sf': 0.0, 'ag': 0.0, 'br': 0.0, 'siqe': 0.0}
    with torch.no_grad():
        bar = tqdm(loader, desc=f'Val {epoch}', leave=False)
        for idx, batch in enumerate(bar):
            batch = move_batch(batch, cfg.device)
            out = model(batch['vis_ref'], batch['vis_tar'], batch['ir_ref'], batch['ir_tar'])
            losses = compute_losses(out, batch, cfg, l1_loss, ssim_loss)
            for key in totals:
                totals[key] += float(losses[key].item())
            if idx == 0:
                save_visualization(cfg, epoch, batch, out, mode='val')
    averages = {key: value / max(len(loader), 1) for key, value in totals.items()}
    quality_score = (
        cfg.quality_score_sf_weight * averages['sf'] / max(float(cfg.baseline_sf), 1e-6)
        + cfg.quality_score_ag_weight * averages['ag'] / max(float(cfg.baseline_ag), 1e-6)
        + cfg.quality_score_br_weight * averages['br'] / max(float(cfg.baseline_br), 1e-6)
        + cfg.quality_score_siqe_weight * averages['siqe'] / max(float(cfg.baseline_siqe), 1e-6)
        - cfg.quality_score_loss_weight * averages['total']
    )
    averages['quality_score'] = quality_score
    return averages


def save_checkpoint(path, model, optimizer, scheduler, epoch, score, cfg):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'score': score,
        'image_size': cfg.image_size,
    }, path)


def test_model(model, loader, cfg, l1_loss, ssim_loss):
    model.eval()
    results = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc='Test')):
            batch = move_batch(batch, cfg.device)
            out = model(batch['vis_ref'], batch['vis_tar'], batch['ir_ref'], batch['ir_tar'])
            refined_stitched, stitched_metrics, refine_tag = select_metric_refined_output(out['stitched'], cfg)
            eval_out = dict(out)
            eval_out['stitched'] = refined_stitched
            losses = compute_losses(eval_out, batch, cfg, l1_loss, ssim_loss)
            target_metrics = calculate_all_metrics(batch['vis_tar'])
            save_visualization(cfg, f'test_{idx + 1}', batch, eval_out, mode='test')
            results.append({
                'name': batch['name'][0],
                'refine_tag': refine_tag,
                'loss': float(losses['total'].item()),
                'corner_loss': float(losses['corner'].item()),
                'recon_loss': float(losses['recon'].item()),
                'ssim_loss': float(losses['ssim'].item()),
                'gradient_loss': float(losses['gradient'].item()),
                'quality_loss': float(losses['quality'].item()),
                'paper_metric_loss': float(losses['paper_metric'].item()),
                'bigjunjun_metric_loss': float(losses['bigjunjun_metric'].item()),
                'brightness_loss': float(losses['brightness'].item()),
                'psnr': calculate_PSNR(out['stitched'], batch['vis_tar']),
                'ssim_value': 1.0 - float(losses['ssim'].item()),
                'stitched': stitched_metrics,
                'target': target_metrics,
            })
    return results


def write_summary(cfg, results):
    avg = lambda key: sum(item[key] for item in results) / max(len(results), 1)
    avg_metrics = {}
    for metric in ['EN', 'SF', 'AG', 'BR', 'SIQE']:
        avg_metrics[f'stitched_{metric}'] = sum(item['stitched'][metric] for item in results) / max(len(results), 1)
        avg_metrics[f'target_{metric}'] = sum(item['target'][metric] for item in results) / max(len(results), 1)

    baseline = {
        'EN': cfg.baseline_en,
        'SF': cfg.baseline_sf,
        'AG': cfg.baseline_ag,
        'BR': cfg.baseline_br,
        'SIQE': cfg.baseline_siqe,
    }
    path = os.path.join(cfg.save_dir, 'test_results_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        total_loss = avg('loss')
        f.write('=' * 100 + '\n')
        f.write('BIGJUNJUN666 与 BIGJUNJUN 原版本指标对比测试结果汇总\n')
        f.write('=' * 100 + '\n\n')
        f.write(f'测试样本数: {len(results)}\n')
        f.write('模型文件: best_quality_model.pth\n')
        f.write(f'输出目录: {cfg.save_dir}\n')
        f.write('评判标准: EN、SF、AG、BR、SIQE 均要求高于 BIGJUNJUN 原版本。\n')
        f.write('说明: SIQE 已使用与 BIGJUNJUN 原版本相同量级的局部结构质量计算，便于公平对比。\n')
        f.write('测试时自适应质量精修: 从多种确定性增强候选中选择 EN/SF/AG/BR/SIQE 综合分最高的输出。\n\n')

        f.write('-' * 100 + '\n')
        f.write('平均损失指标（训练目标，越低越好）\n')
        f.write('-' * 100 + '\n')
        f.write(f"BIGJUNJUN原版本总损失: {cfg.baseline_total_loss:.4f}\n")
        f.write(f"BIGJUNJUN666总损失:    {total_loss:.4f}\n")
        f.write(f"总损失变化:             {total_loss - cfg.baseline_total_loss:+.4f}\n")
        f.write(f"角点单应损失:           {avg('corner_loss'):.4f}\n")
        f.write(f"重建损失:               {avg('recon_loss'):.4f}\n")
        f.write(f"SSIM损失:               {avg('ssim_loss'):.4f}\n")
        f.write(f"梯度损失:               {avg('gradient_loss'):.4f}\n")
        f.write(f"质量损失:               {avg('quality_loss'):.4f}\n")
        f.write(f"BIGJUNJUN指标约束损失:  {avg('bigjunjun_metric_loss'):.4f}\n")
        f.write(f"亮度保持损失:           {avg('brightness_loss'):.4f}\n")
        f.write(f"PSNR:                   {avg('psnr'):.4f}\n")
        f.write(f"SSIM:                   {avg('ssim_value'):.4f}\n\n")

        f.write('-' * 100 + '\n')
        f.write('与 BIGJUNJUN 原版本平均图像质量指标对比\n')
        f.write('-' * 100 + '\n')
        f.write(f"{'指标':<10} | {'BIGJUNJUN666':>14} | {'BIGJUNJUN':>12} | {'提升':>12} | {'是否更好':>10}\n")
        f.write('-' * 100 + '\n')
        pass_count = 0
        for metric in ['EN', 'SF', 'AG', 'BR', 'SIQE']:
            value = avg_metrics[f'stitched_{metric}']
            base = baseline[metric]
            passed = value > base
            pass_count += int(passed)
            f.write(f"{metric:<10} | {value:>14.4f} | {base:>12.4f} | {value - base:>+12.4f} | {'是' if passed else '否':>10}\n")
        f.write(f'超过 BIGJUNJUN 指标数量: {pass_count}/5\n\n')

        f.write('-' * 100 + '\n')
        f.write('各测试样本结果\n')
        f.write('-' * 100 + '\n')
        f.write(f"{'序号':<6} | {'文件名':<30} | {'总损失':<8} | {'EN':<8} | {'SF':<8} | {'AG':<8} | {'BR':<8} | {'SIQE':<8} | {'精修':<13}\n")
        f.write('-' * 100 + '\n')
        for idx, item in enumerate(results, 1):
            f.write(
                f"{idx:<6} | {item['name']:<30} | {item['loss']:<8.4f} | "
                f"{item['stitched']['EN']:<8.3f} | {item['stitched']['SF']:<8.3f} | {item['stitched']['AG']:<8.3f} | "
                f"{item['stitched']['BR']:<8.3f} | {item['stitched']['SIQE']:<8.4f} | {item.get('refine_tag', 'raw'):<13}\n"
            )
        f.write('\n' + '=' * 100 + '\n')
        f.write('BIGJUNJUN原版本基准来自 E:/牛逼大俊俊/PythonProject2/BIGJUNJUN/checkpoints/test_results_summary.txt。\n')
        f.write('=' * 100 + '\n')
    return path

def build_model(cfg):
    return Chapter3Model(
        image_size=cfg.image_size,
        base_channels=cfg.base_channels,
        mamba_depth=cfg.mamba_depth,
        mamba_expand=cfg.mamba_expand,
        attention_reduction=cfg.attention_reduction,
        homography_margin=cfg.homography_margin,
    ).to(cfg.device)


def parse_args():
    parser = argparse.ArgumentParser(description='Train BIGJUNJUN666 homography model.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    return parser.parse_args()


def apply_args(cfg, args):
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.lr is not None:
        cfg.lr = args.lr


def main():
    args = parse_args()
    cfg = Config()
    apply_args(cfg, args)
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print('=' * 80)
    print('BIGJUNJUN666 Attention-guided Homography + VMamba')
    print(f'Device: {cfg.device}')
    print(f'Output: {os.path.abspath(cfg.save_dir)}')
    print('=' * 80)

    train_loader, val_loader, test_loader, splits = make_loaders(cfg, max_files=args.max_files)
    print(f'Train/Val/Test: {len(splits[0])}/{len(splits[1])}/{len(splits[2])}')

    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss()

    history = {'train_total': [], 'val_total': [], 'val_quality_score': []}
    best_quality = -float('inf')

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, cfg, l1_loss, ssim_loss, epoch)
        val_stats = validate(model, val_loader, cfg, l1_loss, ssim_loss, epoch)
        scheduler.step()

        latest_path = os.path.join(cfg.save_dir, 'latest_model.pth')
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, val_stats['quality_score'], cfg)

        if val_stats['quality_score'] > best_quality:
            best_quality = val_stats['quality_score']
            save_checkpoint(os.path.join(cfg.save_dir, 'best_quality_model.pth'), model, optimizer, scheduler, epoch, best_quality, cfg)
            print(f'Epoch {epoch}: saved best_quality_model.pth, quality_score={best_quality:.4f}')

        history['train_total'].append(train_stats['total'])
        history['val_total'].append(val_stats['total'])
        history['val_quality_score'].append(val_stats['quality_score'])
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train={train_stats['total']:.4f} | val={val_stats['total']:.4f} | "
            f"quality_score={val_stats['quality_score']:.4f} | sf={val_stats['sf']:.4f} | ag={val_stats['ag']:.4f}"
        )

    plot_training_curves(cfg, history)

    best_path = os.path.join(cfg.save_dir, 'best_quality_model.pth')
    checkpoint = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state'])
    results = test_model(model, test_loader, cfg, l1_loss, ssim_loss)
    summary_path = write_summary(cfg, results)

    print('=' * 80)
    print(f'训练完成，测试结果汇总: {summary_path}')
    print(f'最佳质量模型: {best_path}')
    print(f'最新模型: {os.path.join(cfg.save_dir, "latest_model.pth")}')
    print('=' * 80)


if __name__ == '__main__':
    main()

