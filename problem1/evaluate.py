import os
import math
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from utils import *
from pathlib import Path
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from utils import generate_anchors

BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
VIS_DIR = RESULT_DIR / "visualizations"
DATA_DIR = PROJ_DIR / "datasets"
RESULT_DIR.mkdir(parents=True, exist_ok=True);
VIS_DIR.mkdir(parents=True, exist_ok=True)

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh = 0.5):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes.t()
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-8)

        rest = rest[iou <= iou_thresh]
        order = rest
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _decode_layer(pred_map: torch.Tensor, anchors_xyxy: torch.Tensor, num_classes):
    B, ch, H, W = pred_map.shape
    A = anchors_xyxy.shape[0] // (H * W)
    K = 5 + num_classes
    assert ch == A * K

    pred = pred_map.permute(0, 2, 3, 1).contiguous().view(B, -1, K)
    t_xywh = pred[..., :4]
    obj_logit = pred[..., 4]
    cls_logit = pred[..., 5:]

    anc = anchors_xyxy.to(pred_map.device)
    ax = (anc[:, 0] + anc[:, 2]) * 0.5
    ay = (anc[:, 1] + anc[:, 3]) * 0.5
    aw = (anc[:, 2] - anc[:, 0]).clamp(min=1e-6)
    ah = (anc[:, 3] - anc[:, 1]).clamp(min=1e-6)

    gx = t_xywh[..., 0] * aw + ax
    gy = t_xywh[..., 1] * ah + ay
    gw = torch.exp(t_xywh[..., 2]) * aw
    gh = torch.exp(t_xywh[..., 3]) * ah

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    obj_prob = torch.sigmoid(obj_logit)
    cls_prob = torch.softmax(cls_logit, dim=-1)
    return boxes, obj_prob, cls_prob


def decode_predictions(predictions, anchors, num_classes, score_thresh = 0.9, nms_thresh = 0.1):

    B = predictions[0].shape[0]
    per_image_results = [[] for _ in range(B)]

    for s, (p_map, anc) in enumerate(zip(predictions, anchors)):
        boxes, obj_prob, cls_prob = _decode_layer(p_map, anc, num_classes)
        B_, N, C = cls_prob.shape

        for b in range(B_):
            cls_scores, cls_labels = cls_prob[b].max(dim=1)
            scores = obj_prob[b] * cls_scores
            keep = scores >= score_thresh
            if keep.sum() == 0:
                continue

            boxes_b = boxes[b][keep]
            scores_b = scores[keep]
            labels_b = cls_labels[keep]


            for c in range(num_classes):
                idx = torch.nonzero(labels_b == c, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                k_boxes = boxes_b[idx]
                k_scores = scores_b[idx]
                keep_idx = nms(k_boxes, k_scores, nms_thresh)

                for i in keep_idx:
                    per_image_results[b].append({
                        "box": k_boxes[i].detach().cpu(),
                        "score": float(k_scores[i].item()),
                        "label": int(c),
                        "scale": int(s),
                    })
    return per_image_results


def compute_ap(predictions, ground_truths, iou_threshold = 0.5):
    gt_flags = {}
    npos = 0
    for img_id, gts in ground_truths.items():
        gts_t = torch.tensor(gts, dtype=torch.float32)
        ground_truths[img_id] = gts_t
        gt_flags[img_id] = torch.zeros((len(gts),), dtype=torch.bool)
        npos += len(gts)

    if npos == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    predictions = sorted(predictions, key=lambda d: d["score"], reverse=True)

    tp = []
    fp = []
    for pred in predictions:
        img_id = pred["image_id"]
        box = torch.tensor(pred["box"], dtype=torch.float32).view(1, 4)

        gts = ground_truths.get(img_id, torch.zeros((0, 4)))
        flags = gt_flags[img_id]

        if gts.numel() == 0:
            tp.append(0)
            fp.append(1)
            continue

        ious = compute_iou(box, gts).squeeze(0)
        max_iou, max_idx = ious.max(dim=0)

        if max_iou >= iou_threshold and not bool(flags[max_idx]):
            tp.append(1)
            fp.append(0)
            flags[max_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / float(npos)
    precision = tp / np.maximum(tp + fp, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap), precision, recall


def visualize_detections(image, predictions, ground_truths, save_path):

    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        img = (img.numpy() * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = np.array(image)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")


    for b in ground_truths:
        x1, y1, x2, y2 = [float(v) for v in b]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="g", facecolor="none")
        ax.add_patch(rect)

    for d in predictions:
        x1, y1, x2, y2 = [float(v) for v in np.array(d["box"]).tolist()]
        score = d.get("score", 0.0)
        label = d.get("label", -1)
        scale = d.get("scale", -1)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, f"c{label} {score:.2f} s{scale}",
                color="r", fontsize=8, bbox=dict(facecolor="white", alpha=0.5, lw=0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def analyze_scale_performance(model, dataloader, anchors,
                              num_classes = 3,
                              score_thresh = 0.9,
                              nms_thresh = 0.1,
                              iou_thresh_match = 0.5,
                              save_dir = None,
                              max_vis = 10):
    if save_dir is None:
        save_dir = str(VIS_DIR)
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    scale_hit = {0: {"S": 0, "M": 0, "L": 0}, 1: {"S": 0, "M": 0, "L": 0}, 2: {"S": 0, "M": 0, "L": 0}}
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs_t = torch.stack(imgs, dim=0).to(device)
            preds = model(imgs_t)

            per_img = decode_predictions(preds, anchors, num_classes,
                                         score_thresh=score_thresh, nms_thresh=nms_thresh)

            B = len(imgs)
            for i in range(B):
                dets = per_img[i]  # list[dict]


                gt_boxes = targets[i]["boxes"].to(device)  # [G,4]
                if gt_boxes.numel() == 0:
                    pass
                else:
                    if len(dets) > 0:
                        det_boxes = torch.stack(
                            [torch.as_tensor(d["box"], dtype=torch.float32, device=device) for d in dets],
                            dim=0
                        )

                        iou_mat = compute_iou(gt_boxes, det_boxes)
                        matched_det = set()
                        for g in range(iou_mat.shape[0]):
                            best_det = int(torch.argmax(iou_mat[g]).item())
                            if iou_mat[g, best_det] >= iou_thresh_match and best_det not in matched_det:
                                matched_det.add(best_det)
                                x1, y1, x2, y2 = gt_boxes[g]
                                side = float(torch.sqrt(torch.clamp((x2 - x1) * (y2 - y1), min=1.0)).item())
                                if side < 48:
                                    tag = "S"
                                elif side < 96:
                                    tag = "M"
                                else:
                                    tag = "L"
                                scale_hit[dets[best_det]["scale"]][tag] += 1


                if vis_count < max_vis:
                    gts = targets[i]["boxes"].tolist()
                    save_path = os.path.join(save_dir, f"det_{batch_idx:03d}_{i}.png")
                    visualize_detections(imgs[i], dets, gts, save_path)
                    vis_count += 1

    labels = ["S", "M", "L"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in range(3):
        y = [scale_hit[s][k] for k in labels]
        ax.bar(x + (s - 1) * width, y, width=width, label=f"scale{s}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("matched detections")
    ax.set_title("Scale specialization (by object size)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scale_specialization.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(save_dir, "scale_stats.json"), "w", encoding="utf-8") as f:
        json.dump(scale_hit, f, ensure_ascii=False, indent=2)

    return scale_hit

def main():


    VAL_IMG_DIR  = str(DATA_DIR / "detection" / "val")
    VAL_ANN_FILE = str(DATA_DIR / "detection" / "val_annotations.json")
    IMAGE_SIZE = 224
    FEATURE_MAP_SIZES = [(56,56), (28,28), (14,14)]
    ANCHOR_SCALES = [[12,16,24], [32,48,64], [96,128,160]]

    ds = ShapeDetectionDataset(VAL_IMG_DIR, VAL_ANN_FILE, transform=T.ToTensor())
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    model.load_state_dict(torch.load(RESULT_DIR / "best_model.pth", map_location=device))
    model.eval()

    anchors = generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SCALES, image_size=IMAGE_SIZE)


    stats = analyze_scale_performance(model, loader, anchors, save_dir=str(VIS_DIR))
    print("scale hits:", stats)


    all_dets, all_tgts = [], []
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs_t = torch.stack(imgs, 0).to(device)
            preds = model(imgs_t)
            per_img = decode_predictions(preds, anchors, num_classes=3)
            all_dets.extend(per_img); all_tgts.extend(tgts)

    for cls_id in range(3):
        preds_cls, gts_cls = [], {}
        for img_id, (dets, tgt) in enumerate(zip(all_dets, all_tgts)):
            for d in dets:
                if d["label"] == cls_id:
                    preds_cls.append({
                        "image_id": img_id,
                        "score": d["score"],
                        "box": (d["box"].tolist() if hasattr(d["box"], "tolist") else list(d["box"]))
                    })
            gts_cls[img_id] = [b for b, l in zip(tgt["boxes"].tolist(), tgt["labels"].tolist()) if l == cls_id]
        ap, _, _ = compute_ap(preds_cls, gts_cls, iou_threshold=0.5)
        print(f"AP(class {cls_id}) = {ap:.4f}")

if __name__ == "__main__":
    main()