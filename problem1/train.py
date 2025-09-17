import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import torchvision.transforms as T
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # problem1/
PROJ_DIR = BASE_DIR.parent                      # 项目根
RESULT_DIR = BASE_DIR / "results"               # problem1/results/
VIS_DIR = RESULT_DIR / "visualizations"         # problem1/results/visualizations
RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
FEATURE_MAP_SIZES = [(56, 56), (28, 28), (14, 14)]
ANCHOR_SCALES = [[12, 16, 24], [32, 48, 64], [96, 128, 160]]

TRAIN_IMG_DIR = "datasets/detection/train"
TRAIN_ANN_FILE = "datasets/detection/train_annotations.json"
VAL_IMG_DIR = "datasets/detection/val"
VAL_ANN_FILE = "datasets/detection/val_annotations.json"


def detection_collate(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    # Training loop
    anchors = generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SCALES, image_size=IMAGE_SIZE)

    total = {"loss_total": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_loc": 0.0}
    n_batches = 0

    for imgs, targets in dataloader:
        imgs = torch.stack(imgs, dim=0).to(device)

        optimizer.zero_grad()
        preds = model(imgs)

        loss_dict = criterion(preds, targets, anchors)
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        for k in total:
            total[k] += float(loss_dict[k].item())
        n_batches += 1

    for k in total:
        total[k] /= max(n_batches, 1)
    return total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    # Validation loop
    model.eval()
    anchors = generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SCALES, image_size=IMAGE_SIZE)

    total = {"loss_total": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_loc": 0.0}
    n_batches = 0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = torch.stack(imgs, dim=0).to(device)
            preds = model(imgs)
            loss_dict = criterion(preds, targets, anchors)

            for k in total:
                total[k] += float(loss_dict[k].item())
            n_batches += 1

    for k in total:
        total[k] /= max(n_batches, 1)
    return total


def main():
    # Configuration
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset, model, loss, optimizer
    # Training loop with logging
    # Save best model and training log
    transform = T.ToTensor()
    train_set = ShapeDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transform=transform)
    val_set = ShapeDetectionDataset(VAL_IMG_DIR, VAL_ANN_FILE, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=detection_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=detection_collate
    )


    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    log = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        log["train"].append({"epoch": epoch, **train_metrics})
        log["val"].append({"epoch": epoch, **val_metrics})

        print(f"[Epoch {epoch:03d}] "
              f"train_total={train_metrics['loss_total']:.4f} "
              f"val_total={val_metrics['loss_total']:.4f}")


        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            torch.save(model.state_dict(), RESULT_DIR / "best_model.pth")


        with open(RESULT_DIR / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

    print("Done. Best val loss:", best_val)


if __name__ == '__main__':
    main()