import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


class MultiModalDataset(Dataset):
    def __init__(
        self,
        vis_ref_dir,
        vis_tar_dir,
        ir_ref_dir,
        ir_tar_dir,
        image_size,
        files,
        homography_margin=24,
    ):
        self.vis_ref_dir = vis_ref_dir
        self.vis_tar_dir = vis_tar_dir
        self.ir_ref_dir = ir_ref_dir
        self.ir_tar_dir = ir_tar_dir
        self.image_size = image_size
        self.files = files
        self.homography_margin = homography_margin

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def _find_modal_file(self, directory, filename):
        base_name = os.path.splitext(filename)[0]
        for ext in IMG_EXTENSIONS:
            candidate = os.path.join(directory, base_name + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f'Cannot find paired file for {filename} in {directory}')

    def load_img(self, path, gray=False):
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flag)
        if img is None:
            raise FileNotFoundError(path)
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

    def random_homography(self):
        width, height = self.image_size
        src = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        jitter = np.random.uniform(
            -self.homography_margin,
            self.homography_margin,
            src.shape,
        ).astype(np.float32)
        dst = src + jitter
        homography = cv2.getPerspectiveTransform(src, dst).astype(np.float32)
        return src, dst, homography, jitter.reshape(-1).astype(np.float32)

    def __getitem__(self, idx):
        filename = self.files[idx]

        vis_ref_path = os.path.join(self.vis_ref_dir, filename)
        vis_tar_path = os.path.join(self.vis_tar_dir, filename)
        ir_ref_path = self._find_modal_file(self.ir_ref_dir, filename)
        ir_tar_path = self._find_modal_file(self.ir_tar_dir, filename)

        vis_ref = self.load_img(vis_ref_path, gray=False)
        vis_tar = self.load_img(vis_tar_path, gray=False)
        ir_ref = self.load_img(ir_ref_path, gray=True)
        ir_tar = self.load_img(ir_tar_path, gray=True)

        src_corners, dst_corners, homography, corner_offsets = self.random_homography()
        vis_ref_warped = cv2.warpPerspective(vis_ref, homography, self.image_size)
        ir_ref_warped = cv2.warpPerspective(ir_ref, homography, self.image_size)

        return {
            'vis_ref': self.transform_rgb(vis_ref_warped),
            'vis_tar': self.transform_rgb(vis_tar),
            'ir_ref': self.transform_gray(ir_ref_warped),
            'ir_tar': self.transform_gray(ir_tar),
            'corner_offsets': torch.from_numpy(corner_offsets),
            'src_corners': torch.from_numpy(src_corners.reshape(-1)),
            'dst_corners': torch.from_numpy(dst_corners.reshape(-1)),
            'homography': torch.from_numpy(homography),
            'name': filename,
        }
