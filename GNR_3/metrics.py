import torch
import torch.nn.functional as F


def _predicted_classes(pred):
    # Convert model output logits into final class labels.
    # Model outputs logits, so class prediction is argmax over channels.
    if pred.ndim == 4:
        return torch.argmax(pred, dim=1)
    return pred.long()


def global_accuracy(pred, target, num_classes):
    # Global accuracy = total correct pixels / total pixels.
    pred = _predicted_classes(pred)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def class_accuracy(pred, target, num_classes):
    # Class accuracy = average accuracy over all classes.
    pred = _predicted_classes(pred)

    accuracies = []
    for class_id in range(num_classes):
        target_mask = target == class_id
        denom = target_mask.sum().item()
        if denom == 0:
            continue
        correct = (pred[target_mask] == class_id).sum().item()
        accuracies.append(correct / denom)

    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def per_class_iou(pred, target, num_classes):
    # Compute IoU for each class separately.
    #IoU = intersection / union for each class, then mean_iou averages these.
    #intersction means how many pixels are correctly predicted for that class, 
    #union counts all pixels that are either in the prediction or the target for that class.
    #higher IoU means better overlap between prediction and ground truth for that class.
    pred = _predicted_classes(pred)

    ious = []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue
        ious.append(intersection / union)

    return ious


def mean_iou(pred, target, num_classes):
    # mIoU = average IoU over all classes.
    #higher mIoU means better overall segmentation performance across all classes. 
    ious = per_class_iou(pred, target, num_classes)
    return sum(ious) / len(ious) if ious else 0.0


def mean_dice(pred, target, num_classes):
    # Dice score shows overlap between prediction and ground truth.
    # higher Dice means better segmentation quality
    pred = _predicted_classes(pred)

    dice_scores = []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        intersection = (pred_mask & target_mask).sum().item()
        total = pred_mask.sum().item() + target_mask.sum().item()

        if total == 0:
            continue
        dice_scores.append((2.0 * intersection) / total)

    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0


def metric_dict(pred, target, num_classes):
    # Return all metrics together so test.py can save them easily.
    # IoU is kept as the same macro-average as mIoU for this multiclass setup.
    miou = mean_iou(pred, target, num_classes)
    return {
        "G": global_accuracy(pred, target, num_classes),
        "C": class_accuracy(pred, target, num_classes),
        "mIoU": miou,
        "IoU": miou,
        "Dice": mean_dice(pred, target, num_classes),
    }
