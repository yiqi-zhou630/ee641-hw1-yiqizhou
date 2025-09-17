import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch, json, os

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from evaluate import analyze_scale_performance, decode_predictions, compute_ap
from utils import generate_anchors


VAL_IMG_DIR = "datasets/detection/val"
VAL_ANN_FILE = "datasets/detection/val_annotations.json"
IMAGE_SIZE = 224
FEATURE_MAP_SIZES = [(56,56), (28,28), (14,14)]
ANCHOR_SCALES = [[12,16,24], [32,48,64], [96,128,160]]


ds = ShapeDetectionDataset(VAL_IMG_DIR, VAL_ANN_FILE, transform=T.ToTensor())
loader = DataLoader(ds, batch_size=8, shuffle=False,
                    collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
model.load_state_dict(torch.load("results/best_model.pt", map_location=device))
model.eval()

anchors = generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SCALES, image_size=IMAGE_SIZE)

all_anchors = torch.cat(anchors, dim=0)

print("anchors shape:", all_anchors.shape)
print("anchors min/max:",
      all_anchors[:, 0].min().item(), all_anchors[:, 1].min().item(),
      all_anchors[:, 2].max().item(), all_anchors[:, 3].max().item())

stats = analyze_scale_performance(model, loader, anchors)
print("scale hits:", stats)


from evaluate import decode_predictions

all_dets = []
all_tgts = []

with torch.no_grad():
    for imgs, tgts in loader:
        imgs_t = torch.stack(imgs, 0).to(device)
        preds = model(imgs_t)
        per_img = decode_predictions(preds, anchors, num_classes=3)
        all_dets.extend(per_img)
        all_tgts.extend(tgts)


for cls_id in range(3):
    preds_cls = []
    gts_cls = {}

    for img_id, (dets, tgt) in enumerate(zip(all_dets, all_tgts)):
        for d in dets:
            if d["label"] == cls_id:
                preds_cls.append({
                    "image_id": img_id,
                    "score": d["score"],
                    "box": (d["box"].tolist() if hasattr(d["box"], "tolist") else list(d["box"]))
                })

        gts_cls[img_id] = [
            b for b, l in zip(tgt["boxes"].tolist(), tgt["labels"].tolist())
            if l == cls_id
        ]

    ap, prec, rec = compute_ap(preds_cls, gts_cls, iou_threshold=0.5)
    print(f"AP(class {cls_id}) = {ap:.4f}")

