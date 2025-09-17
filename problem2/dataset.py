import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
VIS_DIR = RESULT_DIR / "visualizations"
DATA_DIR = PROJ_DIR / "datasets"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

_NAMES = ["head", "left_hand", "right_hand", "left_foot", "right_foot"]

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap',
                 heatmap_size=64, sigma=2.0):
        """
        Initialize the keypoint dataset.

        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        # Load annotations
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)


        id2name = {}
        annos = None

        if isinstance(data, dict):
            if "images" in data and isinstance(data["images"], list):
                for img in data["images"]:
                    if "id" in img and "file_name" in img:
                        id2name[img["id"]] = img["file_name"]

            if "annotations" in data and isinstance(data["annotations"], list):
                annos = data["annotations"]
            elif "images" in data and isinstance(data["images"], list) and \
                    len(data["images"]) > 0 and "keypoints" in data["images"][0]:
                annos = data["images"]

        elif isinstance(data, list):
            annos = data



        self.items = []
        for a in annos:
            if "file_name" in a:
                fname = a["file_name"]
            else:
                fname = id2name.get(a.get("image_id"))
            if not fname:
                continue

            kps = self._parse_keypoints(a.get("keypoints"))
            if kps is None:
                continue

            self.items.append({
                "file": fname,
                "kps": kps.astype(np.float32)
            })

        self.img_h = 128
        self.img_w = 128

    def _parse_keypoints(self, kp_raw):
        if kp_raw is None:
            return None

        if isinstance(kp_raw, dict):
            out = []
            for name in _NAMES:
                v = kp_raw.get(name)
                if v is None or len(v) < 2:
                    return None
                out.append([float(v[0]), float(v[1])])
            return np.array(out, dtype=np.float32)


        arr = np.array(kp_raw, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size != 10:
                return None
            arr = arr.reshape(5, 2)
        elif arr.ndim == 2:
            if arr.shape != (5, 2):
                return None
        else:
            return None
        return arr

    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.

        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap

        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        # For each keypoint:
        # 1. Create 2D gaussian centered at keypoint location
        # 2. Handle boundary cases
        K = keypoints.shape[0]
        heatmaps = torch.zeros((K, height, width), dtype=torch.float32)

        sx = (width / self.img_w)
        sy = (height / self.img_h)
        kps_hm = keypoints.copy()
        kps_hm[:, 0] *= sx
        kps_hm[:, 1] *= sy


        yy = torch.arange(height, dtype=torch.float32).view(-1, 1).expand(height, width)
        xx = torch.arange(width, dtype=torch.float32).view(1, -1).expand(height, width)

        s2 = 2 * (self.sigma ** 2)
        for i in range(K):
            x0, y0 = float(kps_hm[i, 0]), float(kps_hm[i, 1])

            if not (0 <= x0 < width and 0 <= y0 < height):
                continue
            g = torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / s2)
            heatmaps[i] = g

        return heatmaps

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [5, 64, 64] (5 heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [10] (x,y for 5 keypoints, normalized to [0,1])
        """
        rec = self.items[idx]

        img_path = os.path.join(self.image_dir, rec["file"])
        with Image.open(img_path) as im:
            im = im.convert("L").resize((self.img_w, self.img_h), Image.BILINEAR)
            img = np.array(im, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        kps = rec["kps"]

        if self.output_type == "heatmap":
            target = self.generate_heatmap(kps, self.heatmap_size, self.heatmap_size)
            return img, target
        else:

            norm_x = kps[:, 0] / (self.img_w - 1)
            norm_y = kps[:, 1] / (self.img_h - 1)
            out = np.stack([norm_x, norm_y], axis=1).reshape(-1)
            target = torch.from_numpy(out.astype(np.float32))  # [10]
            return img, target