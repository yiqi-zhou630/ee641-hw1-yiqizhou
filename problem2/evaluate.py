import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
VIS_DIR = RESULT_DIR / "visualizations"
DATA_DIR = PROJ_DIR / "datasets"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

def extract_keypoints_from_heatmaps(heatmaps, orig_size=None):
    """
    Extract (x, y) coordinates from heatmaps.

    Args:
        heatmaps: Tensor of shape [batch, num_keypoints, H, W]

    Returns:
        coords: Tensor of shape [batch, num_keypoints, 2]
    """
    # Find argmax location in each heatmap
    # Convert to (x, y) coordinates
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()

    if orig_size is not None:
        scale_x = orig_size / float(W)
        scale_y = orig_size / float(H)
        xs = xs * scale_x
        ys = ys * scale_y

    coords = torch.stack([xs, ys], dim=-1)
    return coords

def _bbox_diag(gt_xy):
    x_min, y_min = gt_xy.min(dim=0).values
    x_max, y_max = gt_xy.max(dim=0).values
    return torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) + 1e-6

def _torso_len(gt_xy):
    head = gt_xy[0]
    feet_mid = (gt_xy[3] + gt_xy[4]) / 2.0
    return torch.norm(head - feet_mid) + 1e-6

def _maybe_to_pixels(xy, img_size=128):
    if torch.max(xy) <= 1.0001:
        return xy * (img_size - 1)
    return xy

def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    """
    Compute PCK at various thresholds.

    Args:
        predictions: Tensor of shape [N, num_keypoints, 2]
        ground_truths: Tensor of shape [N, num_keypoints, 2]
        thresholds: List of threshold values (as fraction of normalization)
        normalize_by: 'bbox' for bounding box diagonal, 'torso' for torso length

    Returns:
        pck_values: Dict mapping threshold to accuracy
    """
    # For each threshold:
    # Count keypoints within threshold distance of ground truth
    img_size = 128
    preds = _maybe_to_pixels(predictions.clone().detach(), img_size)
    gts = _maybe_to_pixels(ground_truths.clone().detach(), img_size)

    N, K, _ = gts.shape
    dists = torch.norm(preds - gts, dim=-1)

    norms = []
    for i in range(N):
        if normalize_by == 'bbox':
            norms.append(_bbox_diag(gts[i]))
        elif normalize_by == 'torso':
            norms.append(_torso_len(gts[i]))

    norms = torch.stack(norms, dim=0).unsqueeze(1)

    ndists = dists / norms

    pck = {}
    for t in thresholds:
        acc = (ndists <= t).float().mean().item()
        pck[t] = acc
    return pck


def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    """
    Plot PCK curves comparing both methods.
    """
    def _xy(pck_dict):
        thrs = sorted(pck_dict.keys())
        vals = [pck_dict[t] for t in thrs]
        return thrs, vals

    th_hm, v_hm = _xy(pck_heatmap)
    th_rg, v_rg = _xy(pck_regression)

    plt.figure(figsize=(5.5, 4.0))
    plt.plot(th_hm, v_hm, marker='o', label='Heatmap')
    plt.plot(th_rg, v_rg, marker='s', label='Regression')
    plt.xlabel('Threshold (fraction of normalization)')
    plt.ylabel('PCK')
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    """
    Visualize predicted and ground truth keypoints on image.
    """
    point_size = 30
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
    else:
        img = np.array(image)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)

    pred = pred_keypoints.detach().cpu().numpy() if isinstance(pred_keypoints, torch.Tensor) else np.asarray(
        pred_keypoints)
    gt = gt_keypoints.detach().cpu().numpy() if isinstance(gt_keypoints, torch.Tensor) else np.asarray(gt_keypoints)

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.scatter(gt[:, 0], gt[:, 1], s=point_size, marker='o', facecolors='none', edgecolors='lime', linewidths=2,
                label='GT')
    plt.scatter(pred[:, 0], pred[:, 1], s=point_size, marker='x', c='red', linewidths=2, label='Pred')
    plt.legend(loc='lower right')
    plt.axis('off')
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    IMG_SIZE = 128
    BATCH_SIZE = 32
    NUM_KEYPOINTS = 5
    THRS = [0.02, 0.05, 0.1, 0.2]
    NORMALIZE_BY = "bbox"

    IMAGE_DIR = str(DATA_DIR / "keypoints" / "val")
    ANNO_FILE = str(DATA_DIR / "keypoints" / "val_annotations.json")
    HEATMAP_CKPT = RESULT_DIR / "heatmap_model.pth"
    REG_CKPT = RESULT_DIR / "regression_model.pth"
    OUT_PCK = VIS_DIR / "pck.png"
    NUM_VIS = 6

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_val = KeypointDataset(
        IMAGE_DIR, ANNO_FILE,
        output_type="regression",
        heatmap_size=64
    )
    loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    hm = HeatmapNet(num_keypoints=NUM_KEYPOINTS).to(device)
    rg = RegressionNet(num_keypoints=NUM_KEYPOINTS).to(device)

    if not HEATMAP_CKPT.exists() or not REG_CKPT.exists():
        raise FileNotFoundError('')

    hm.load_state_dict(torch.load(HEATMAP_CKPT, map_location=device))
    rg.load_state_dict(torch.load(REG_CKPT, map_location=device))
    hm.eval(); rg.eval()

    preds_hm, preds_rg, gts_px, imgs_cache = [], [], [], []
    with torch.no_grad():
        for imgs, tgt_reg01 in loader:
            imgs = imgs.to(device)

            hm_out = hm(imgs)
            preds_hm.append(extract_keypoints_from_heatmaps(hm_out.cpu(), orig_size=IMG_SIZE))

            rg_out01 = rg(imgs).view(-1, NUM_KEYPOINTS, 2).cpu()
            preds_rg.append(rg_out01 * (IMG_SIZE - 1))

            gts_px.append(tgt_reg01.view(-1, NUM_KEYPOINTS, 2) * (IMG_SIZE - 1))
            imgs_cache.append(imgs.cpu())

    preds_hm = torch.cat(preds_hm, dim=0)
    preds_rg = torch.cat(preds_rg, dim=0)
    gts_px = torch.cat(gts_px, dim=0)
    imgs_all = torch.cat(imgs_cache, dim=0)

    pck_hm = compute_pck(preds_hm, gts_px, THRS, normalize_by=NORMALIZE_BY)
    pck_rg = compute_pck(preds_rg, gts_px, THRS, normalize_by=NORMALIZE_BY)
    plot_pck_curves(pck_hm, pck_rg, str(OUT_PCK))
    print("PCK (heatmap):", pck_hm)
    print("PCK (regress):", pck_rg)
    print(f"PCK curve saved")

    idxs = torch.linspace(0, len(ds_val) - 1, steps=NUM_VIS).round().long().tolist()
    for i, idx in enumerate(idxs):
        visualize_predictions(
            image=imgs_all[idx, 0],
            pred_keypoints=preds_hm[idx],
            gt_keypoints=gts_px[idx],
            save_path=str(VIS_DIR / f"hm_pred_{i}.png"),
        )
        visualize_predictions(
            image=imgs_all[idx, 0],
            pred_keypoints=preds_rg[idx],
            gt_keypoints=gts_px[idx],
            save_path=str(VIS_DIR / f"reg_pred_{i}.png"),
        )
    print(f"Sample visualizations saved")

if __name__ == "__main__":
    main()
