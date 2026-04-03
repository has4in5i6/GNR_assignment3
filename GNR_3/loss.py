import torch
import torch.nn as nn
import torch.nn.functional as F

# dice loss represents the overlap between predicted and true segmentation masks,
# higher dice means better segmentation quality, especially for smaller objects.
# Dice = (2 * intersection) / (total pixels in prediction + total pixels in target) for each class, then averaged.
def multiclass_dice_loss(logits, target, num_classes):
    smooth = 1e-6
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice_per_class = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice_per_class.mean()


def combined_loss(logits, target, num_classes, class_weights=None):
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = multiclass_dice_loss(logits, target, num_classes)
    return ce + dice
