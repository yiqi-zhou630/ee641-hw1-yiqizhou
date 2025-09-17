import os, json, math
from copy import deepcopy
from dataclasses import dataclass, asdict
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from model import Encoder
from evaluate import (
    extract_keypoints_from_heatmaps,
    compute_pck,
    plot_pck_curves,
    visualize_predictions,
)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
VIS_DIR = RESULT_DIR / "visualizations"
DATA_DIR = PROJ_DIR / "datasets"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Cfg:
    train_img_dir: str = "datasets/keypoints/train"
    train_ann: str = "datasets/keypoints/train_annotations.json"
    val_img_dir: str = "datasets/keypoints/val"
    val_ann: str = "datasets/keypoints/val_annotations.json"
    img_size: int = 128
    batch_size: int = 32
    workers: int = 2
    epochs: int = 3
    lr: float = 1e-3
    thrs: tuple = (0.02, 0.05, 0.1, 0.2)
    normalize_by: str = "bbox"
    out_dir: str = str(RESULT_DIR / "baseline")


cfg = Cfg()
os.makedirs(cfg.out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 5


def run_train_val(heatmap_size=64, sigma=2.0, use_skip=True, epochs=None):
    train_set = KeypointDataset(cfg.train_img_dir, cfg.train_ann,
                                output_type="heatmap", heatmap_size=heatmap_size, sigma=sigma)
    val_set = KeypointDataset(cfg.val_img_dir, cfg.val_ann,
                              output_type="heatmap", heatmap_size=heatmap_size, sigma=sigma)
    tr_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.workers, pin_memory=True)
    va_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.workers, pin_memory=True)

    if use_skip:
        model = HeatmapNet(num_keypoints=K).to(device)
    else:
        model = HeatmapNoSkip(num_keypoints=K).to(device)

    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    E = epochs or cfg.epochs

    best_state = None
    best_val = float("inf")

    for ep in range(1, E + 1):
        model.train()
        tl_sum, n_sum = 0.0, 0
        for imgs, target in tr_loader:
            imgs, target = imgs.to(device), target.to(device)
            pred = model(imgs)
            if pred.shape[-2:] != target.shape[-2:]:
                pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            loss = crit(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = imgs.size(0);
            tl_sum += float(loss.detach()) * bs;
            n_sum += bs
        tl = tl_sum / max(n_sum, 1)

        model.eval()
        vl_sum, n_sum = 0.0, 0
        with torch.no_grad():
            for imgs, target in va_loader:
                imgs, target = imgs.to(device), target.to(device)
                pred = model(imgs)
                if pred.shape[-2:] != target.shape[-2:]:
                    pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
                loss = crit(pred, target)
                bs = imgs.size(0);
                vl_sum += float(loss) * bs;
                n_sum += bs
        vl = vl_sum / max(n_sum, 1)
        if vl < best_val:
            best_val = vl
            best_state = deepcopy(model.state_dict())

        print(f"[heatmap {heatmap_size}/{sigma}/skip={use_skip}] "
              f"Epoch {ep:02d}/{E}  train {tl:.4f}  val {vl:.4f}")

    model.load_state_dict(best_state)
    preds, gts = [], []
    with torch.no_grad():
        for imgs, _ in va_loader:
            imgs = imgs.to(device)
            hm = model(imgs).cpu()
            preds.append(extract_keypoints_from_heatmaps(hm, orig_size=cfg.img_size))

    val_reg = KeypointDataset(cfg.val_img_dir, cfg.val_ann,
                              output_type="regression", heatmap_size=heatmap_size, sigma=sigma)
    reg_loader = DataLoader(val_reg, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True)
    for _, tgt in reg_loader:
        gts.append(tgt.view(-1, K, 2) * (cfg.img_size - 1))

    preds = torch.cat(preds, 0)
    gts = torch.cat(gts, 0)
    pck = compute_pck(preds, gts, list(cfg.thrs), normalize_by=cfg.normalize_by)
    return pck


class HeatmapNoSkip(nn.Module):

    def __init__(self, num_keypoints=5):
        super().__init__()
        self.enc = Encoder()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_keypoints, kernel_size=1)

    def forward(self, x):
        e1, e2, e3, e4 = self.enc(x)
        u4 = self.deconv4(e4)
        u3 = self.deconv3(u4)
        u2 = self.deconv2(u3)
        return self.head(u2)


def ablation_study():
    results = {
        "config": asdict(cfg),
        "experiments": []
    }

    for hm_size in (32, 64, 128):
        pck = run_train_val(heatmap_size=hm_size, sigma=2.0, use_skip=True)
        results["experiments"].append({
            "tag": f"heatmap_size_{hm_size}",
            "pck": pck
        })

    for s in (1.0, 2.0, 3.0, 4.0):
        pck = run_train_val(heatmap_size=64, sigma=float(s), use_skip=True)
        results["experiments"].append({
            "tag": f"sigma_{s}",
            "pck": pck
        })

    for flag in (True, False):
        pck = run_train_val(heatmap_size=64, sigma=2.0, use_skip=flag)
        results["experiments"].append({
            "tag": f"skip_{int(flag)}",
            "pck": pck
        })

    with open(os.path.join(cfg.out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Ablation results saved", os.path.join(cfg.out_dir, "ablation_results.json"))

    hm_size_pcks = [e for e in results["experiments"] if e["tag"].startswith("heatmap_size")]
    if hm_size_pcks:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.2, 4))
        for e in hm_size_pcks:
            ths = sorted(e["pck"].keys())
            vals = [e["pck"][t] for t in ths]
            plt.plot(ths, vals, marker="o", label=e["tag"])
        plt.xlabel("Threshold");
        plt.ylabel("PCK");
        plt.ylim(0, 1);
        plt.grid(True, ls="--", alpha=.4)
        plt.legend();
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "ablation_heatmap_size.png"), dpi=150)
        plt.close()
        print("Saved", os.path.join(cfg.out_dir, "ablation_heatmap_size.png"))

        sigma_pcks = [e for e in results["experiments"] if e["tag"].startswith("sigma_")]
        if sigma_pcks:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5.2, 4))
            for e in sigma_pcks:
                ths = sorted(e["pck"].keys())
                vals = [e["pck"][t] for t in ths]
                label = e["tag"].replace("sigma_", "sigma=")
                plt.plot(ths, vals, marker="o", label=label)
            plt.xlabel("Threshold");
            plt.ylabel("PCK")
            plt.ylim(0, 1);
            plt.grid(True, ls="--", alpha=.4)
            plt.legend();
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_dir, "ablation_sigma.png"), dpi=150)
            plt.close()
            print("Saved", os.path.join(cfg.out_dir, "ablation_sigma.png"))


def analyze_failure_cases():

    vis_dir = os.path.join(cfg.out_dir, "failures")
    os.makedirs(vis_dir, exist_ok=True)

    val_set = KeypointDataset(cfg.val_img_dir, cfg.val_ann,
                              output_type="regression", heatmap_size=64, sigma=2.0)
    loader = DataLoader(val_set, batch_size=32, shuffle=False,
                        num_workers=cfg.workers, pin_memory=True)

    hm = HeatmapNet(num_keypoints=K).to(device)
    rg = RegressionNet(num_keypoints=K).to(device)
    hm.load_state_dict(torch.load(RESULT_DIR / "heatmap_model.pth", map_location=device))
    rg.load_state_dict(torch.load(RESULT_DIR / "regression_model.pth", map_location=device))
    hm.eval();
    rg.eval()

    preds_hm, preds_rg, gts_px, ims = [], [], [], []
    with torch.no_grad():
        for imgs, tgt01 in loader:
            imgs = imgs.to(device)
            hm_out = hm(imgs).cpu()
            preds_hm.append(extract_keypoints_from_heatmaps(hm_out, orig_size=cfg.img_size))
            preds_rg.append(rg(imgs).view(-1, K, 2).cpu() * (cfg.img_size - 1))
            gts_px.append(tgt01.view(-1, K, 2) * (cfg.img_size - 1))
            ims.append(imgs.cpu())

    preds_hm = torch.cat(preds_hm, 0)
    preds_rg = torch.cat(preds_rg, 0)
    gts_px = torch.cat(gts_px, 0)
    ims = torch.cat(ims, 0)

    def bbox_diag(gt_xy):
        x_min, _ = gt_xy[:, 0].min(dim=0)
        y_min = gt_xy[:, 1].min(dim=0).values
        x_max = gt_xy[:, 0].max(dim=0).values
        y_max = gt_xy[:, 1].max(dim=0).values
        return math.sqrt(float((x_max - x_min) ** 2 + (y_max - y_min) ** 2)) + 1e-6

    nd_hm, nd_rg = [], []
    for i in range(gts_px.shape[0]):
        norm = bbox_diag(gts_px[i])
        d_hm = torch.norm(preds_hm[i] - gts_px[i], dim=-1).mean().item() / norm
        d_rg = torch.norm(preds_rg[i] - gts_px[i], dim=-1).mean().item() / norm
        nd_hm.append(d_hm);
        nd_rg.append(d_rg)

    nd_hm = torch.tensor(nd_hm);
    nd_rg = torch.tensor(nd_rg)
    thr = 0.05
    ok_hm = nd_hm <= thr
    ok_rg = nd_rg <= thr

    idx_hm_ok_rg_bad = torch.where(ok_hm & (~ok_rg))[0][:6]
    idx_rg_ok_hm_bad = torch.where(ok_rg & (~ok_hm))[0][:6]
    idx_both_bad = torch.where((~ok_rg) & (~ok_hm))[0][:6]

    def dump_cases(idxs, tag):
        for j, idx in enumerate(idxs.tolist()):
            visualize_predictions(
                image=ims[idx, 0],
                pred_keypoints=preds_hm[idx],
                gt_keypoints=gts_px[idx],
                save_path=os.path.join(vis_dir, f"{tag}_HM_{j}.png"),
            )
            visualize_predictions(
                image=ims[idx, 0],
                pred_keypoints=preds_rg[idx],
                gt_keypoints=gts_px[idx],
                save_path=os.path.join(vis_dir, f"{tag}_RG_{j}.png"),
            )

    dump_cases(idx_hm_ok_rg_bad, "HM_success_RG_fail")
    dump_cases(idx_rg_ok_hm_bad, "RG_success_HM_fail")
    dump_cases(idx_both_bad, "Both_fail")
    print("Failure cases saved to", vis_dir)


if __name__ == "__main__":
    print("Device:", device)
    ablation_study()
    analyze_failure_cases()
