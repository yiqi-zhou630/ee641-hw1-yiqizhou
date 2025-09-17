import os
import json

import torch
from torch.utils.data import Dataset
from PIL import Image


class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)


        id_to_fname = {
            img["id"]: img["file_name"] for img in data.get("images", [])
        }

        grouped = {}
        for anno in data.get("annotations", []):
            img_id = anno["image_id"]
            bbox = anno["bbox"]
            label = anno["category_id"]
            if img_id not in grouped:
                grouped[img_id] = {"boxes": [], "labels": []}
            grouped[img_id]["boxes"].append(bbox)
            grouped[img_id]["labels"].append(label)


        self.samples = []
        for img_id, fname in id_to_fname.items():
            img_path = os.path.join(self.image_dir, fname)
            anno = grouped.get(img_id, {"boxes": [], "labels": []})
            self.samples.append((img_path, anno["boxes"], anno["labels"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        img_path, boxes_list, labels_list = self.samples[idx]


        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if len(boxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            xyxy = []
            for x1, y1, x2, y2 in boxes_list:
                xyxy.append([x1, y1, x2, y2])
            boxes = torch.tensor(xyxy, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.long)

        targets = {"boxes": boxes, "labels": labels}
        return img, targets