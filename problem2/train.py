import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from dataclasses import dataclass, asdict
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
VIS_DIR = RESULT_DIR / "visualizations"
DATA_DIR = PROJ_DIR / "datasets"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    image_dir = "datasets/keypoints/train"
    anno_file = "datasets/keypoints/train_annotations.json"
    batch_size = 32
    num_workers = 2
    epochs = 30
    lr = 1e-3
    heatmap_size = 64
    sigma = 2.0
    save_dir = str(RESULT_DIR)

cfg = Config()

def make_loaders(output_type):
    train_set = KeypointDataset(
        cfg.image_dir,
        cfg.anno_file,
        output_type=output_type,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma
    )
    val_set = KeypointDataset(
        "datasets/keypoints/val",
        "datasets/keypoints/val_annotations.json",
        output_type=output_type,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader



def run_one_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_cnt = 0, 0

    torch.set_grad_enabled(is_train)
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bs = imgs.size(0)
        total_loss += float(loss.detach()) * bs
        total_cnt += bs

    return total_loss / max(total_cnt, 1)

def train_heatmap_model(model, train_loader, val_loader, num_epochs, device = 'cuda'):
    """
    Train the heatmap-based model.

    Uses MSE loss between predicted and target heatmaps.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    # Log losses and save best model
    best_val = float("inf")
    log = {"type": "heatmap", "cfg": asdict(cfg), "epochs": []}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ep in range(1, num_epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_one_epoch(model, val_loader, criterion, optimizer=None, device=device)

        log["epochs"].append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[Heatmap]Epoch {ep} train_loss = {train_loss:.4f} val_loss = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "heatmap_model.pth"))

    return log


def train_regression_model(model, train_loader, val_loader, num_epochs, device = 'cuda'):
    """
    Train the direct regression model.

    Uses MSE loss between predicted and target coordinates.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    # Log losses and save best model
    best_val = float("inf")
    log = {"type": "regression", "cfg": asdict(cfg), "epochs": []}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ep in range(1, num_epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, optimizer, device = device)
        val_loss = run_one_epoch(model, val_loader, criterion, optimizer=None, device=device)

        log["epochs"].append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[Regress] Epoch = {ep} train = {train_loss:.4f} val = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "regression_model.pth"))

    return log

def main():
    # Train both models with same data
    # Save training logs for comparison
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    hm_train, hm_val = make_loaders(output_type="heatmap")
    heatmap_model = HeatmapNet(num_keypoints=5).to(device)
    log_hm = train_heatmap_model(
        heatmap_model, hm_train, hm_val, num_epochs=cfg.epochs, device=device
    )

    reg_train, reg_val = make_loaders(output_type="regression")
    regression_model = RegressionNet(num_keypoints=5).to(device)
    log_reg = train_regression_model(
        regression_model, reg_train, reg_val, num_epochs=cfg.epochs, device=device
    )

    logs = {"heatmap": log_hm, "regression": log_reg}
    with open(os.path.join(cfg.save_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print("Done. Models saved as 'heatmap_model.pth' and 'regression_model.pth'.")
    print("Logs saved to 'training_log.json'.")


if __name__ == '__main__':
    main()