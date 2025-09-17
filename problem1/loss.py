import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets


def encode_boxes(anchors_xyxy: torch.Tensor, boxes_xyxy: torch.Tensor):
    ax = (anchors_xyxy[:, 0] + anchors_xyxy[:, 2]) * 0.5
    ay = (anchors_xyxy[:, 1] + anchors_xyxy[:, 3]) * 0.5
    aw = (anchors_xyxy[:, 2] - anchors_xyxy[:, 0]).clamp(min=1e-6)
    ah = (anchors_xyxy[:, 3] - anchors_xyxy[:, 1]).clamp(min=1e-6)

    gx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
    gy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5
    gw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1e-6)
    gh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1e-6)

    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=1)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.w_obj = 1.0
        self.w_cls = 1.0
        self.w_loc = 2.0

    def forward(self, predictions, targets, anchors):
        """
        Compute multi-task loss.

        Args:
            predictions: List of tensors from each scale
            targets: List of dicts with 'boxes' and 'labels' for each image
            anchors: List of anchor tensors for each scale

        Returns:
            loss_dict: Dict containing:
                - loss_obj: Objectness loss
                - loss_cls: Classification loss  
                - loss_loc: Localization loss
                - loss_total: Weighted sum
        """
        # For each prediction scale:
        # 1. Match anchors to targets
        # 2. Compute objectness loss (BCE)
        # 3. Compute classification loss (CE) for positive anchors
        # 4. Compute localization loss (Smooth L1) for positive anchors
        # 5. Apply hard negative mining (3:1 ratio)
        device = predictions[0].device
        B = predictions[0].shape[0]
        C = self.num_classes
        K = 5 + C

        total_loc = 0.0
        total_cls = 0.0
        total_obj = 0.0
        total_pos = 0
        total_sel = 0


        for p_map, anc in zip(predictions, anchors):
            bsz, ch, H, W = p_map.shape
            A = anc.shape[0] // (H * W)



            pred = p_map.permute(0, 2, 3, 1).contiguous().view(bsz, -1, K)
            pred_box = pred[..., :4]
            pred_obj_logit = pred[..., 4]
            pred_cls_logit = pred[..., 5:]
            N = pred.shape[1]


            for b in range(B):
                t = targets[b]
                gt_boxes = t["boxes"].to(device)
                gt_labels = t["labels"].to(device)


                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anc.to(device), gt_boxes, gt_labels, pos_threshold=0.5, neg_threshold=0.3
                )

                num_pos = int(pos_mask.sum().item())
                total_pos += num_pos


                obj_target = torch.zeros((N,), dtype=torch.float32, device=device)
                obj_target[pos_mask] = 1.0

                # 先算所有 anchor 的 BCE（不做 reduce），再进行 hard negative mining 挑负样本
                obj_loss_all = F.binary_cross_entropy_with_logits(
                    pred_obj_logit[b], obj_target, reduction="none"
                )


                sel_neg_mask = self.hard_negative_mining(
                    obj_loss_all.detach(), pos_mask, neg_mask, ratio=3
                )

                select_mask = pos_mask | sel_neg_mask
                total_sel += int(select_mask.sum().item())
                total_obj += obj_loss_all[select_mask].sum()

                if num_pos > 0:
                    cls_target = (matched_labels[pos_mask] - 1)  # [0..C-1]
                    cls_loss = F.cross_entropy(
                        pred_cls_logit[b, pos_mask], cls_target, reduction="sum"
                    )
                    total_cls += cls_loss


                    target_t = encode_boxes(anc[pos_mask], matched_boxes[pos_mask])
                    loc_loss = F.smooth_l1_loss(
                        pred_box[b, pos_mask], target_t, reduction="sum"
                    )
                    total_loc += loc_loss

        denom_pos = max(total_pos, 1)
        denom_obj = max(total_sel, 1)

        loss_loc = total_loc / denom_pos
        loss_cls = total_cls / denom_pos
        loss_obj = total_obj / denom_obj

        loss_total = self.w_loc * loss_loc + self.w_cls * loss_cls + self.w_obj * loss_obj

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_loc": loss_loc,
            "loss_total": loss_total,
        }

    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples.

        Args:
            loss: Loss values for all anchors
            pos_mask: Boolean mask for positive anchors
            neg_mask: Boolean mask for negative anchors
            ratio: Negative to positive ratio

        Returns:
            selected_neg_mask: Boolean mask for selected negatives
        """
        num_pos = int(pos_mask.sum().item())
        if num_pos == 0:

            k = max(int(neg_mask.sum().item() // 10), 1) if neg_mask.any() else 0
        else:
            k = min(int(ratio * num_pos), int(neg_mask.sum().item()))

        selected = torch.zeros_like(neg_mask, dtype=torch.bool)
        if k > 0:
            neg_losses = loss.clone()
            neg_losses[~neg_mask] = -1e9
            topk_vals, topk_idx = torch.topk(neg_losses, k=k, dim=0)
            selected[topk_idx] = True
        return selected