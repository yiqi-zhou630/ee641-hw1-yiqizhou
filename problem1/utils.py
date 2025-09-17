import torch
import numpy as np


def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.

    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size

    Returns:
        anchors: List of tensors, each of shape [num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    anchors_per_fmap = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        stride_y = image_size / float(H)
        stride_x = image_size / float(W)

        ys = (torch.arange(H, dtype=torch.float32) + 0.5) * stride_y
        xs = (torch.arange(W, dtype=torch.float32) + 0.5) * stride_x
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([cx, cy], dim=-1).view(-1, 2)

        all_boxes = []
        for s in scales:
            s = float(s)
            half = s / 2.0
            x1y1 = centers - half
            x2y2 = centers + half
            boxes = torch.cat([x1y1[:, 0:1], x1y1[:, 1:2],
                               x2y2[:, 0:1], x2y2[:, 1:2]], dim=1)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, image_size - 1.0)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, image_size - 1.0)
            all_boxes.append(boxes)


        stack = torch.stack(all_boxes, dim=1)
        fmap_anchors = stack.reshape(-1, 4)
        anchors_per_fmap.append(fmap_anchors)
    return anchors_per_fmap


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]

    Returns:
        iou: Tensor of shape [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)


    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]

    x1 = torch.maximum(b1[..., 0], b2[..., 0])
    y1 = torch.maximum(b1[..., 1], b2[..., 1])
    x2 = torch.minimum(b1[..., 2], b2[..., 2])
    y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return iou


def match_anchors_to_targets(anchors, target_boxes, target_labels,
                             pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.

    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors

    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    A = anchors.shape[0]
    device = anchors.device


    matched_labels = torch.zeros((A,), dtype=torch.long, device=device)
    matched_boxes = torch.zeros((A, 4), dtype=torch.float32, device=device)
    pos_mask = torch.zeros((A,), dtype=torch.bool, device=device)
    neg_mask = torch.zeros((A,), dtype=torch.bool, device=device)


    if target_boxes is None or target_boxes.numel() == 0:
        neg_mask[:] = True
        return matched_labels, matched_boxes, pos_mask, neg_mask


    iou = compute_iou(anchors, target_boxes)
    best_iou, best_gt_idx = iou.max(dim=1)


    pos_mask = best_iou >= pos_threshold
    neg_mask = best_iou < neg_threshold


    gt_best_iou, gt_best_anchor = iou.max(dim=0)  # [T]
    pos_mask[gt_best_anchor] = True

    best_gt_idx[gt_best_anchor] = torch.arange(target_boxes.shape[0], device=anchors.device)
    neg_mask[gt_best_anchor] = False

    if pos_mask.any():
        matched_boxes[pos_mask] = target_boxes[best_gt_idx[pos_mask]]
        matched_labels[pos_mask] = target_labels[best_gt_idx[pos_mask]] + 1

    return matched_labels, matched_boxes, pos_mask, neg_mask