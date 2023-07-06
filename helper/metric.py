import torch
import torch.nn.functional as F


def dice_score(prediction, ground_truth, smooth=1e-9):
    intersection = 2. * torch.sum(prediction * ground_truth, dim=(1, 2))
    xy = prediction.sum(dim=(1, 2)) + ground_truth.sum(dim=(1, 2))
    return torch.div(intersection, xy + smooth)


def DiceLoss(prediction, ground_truth):
    sig_pred = F.sigmoid(prediction)
    non_zero_bool = (ground_truth.sum(dim=(2, 3)) > 0) | (sig_pred.sum(dim=(2, 3)) > 0)
    non_zero_idx = torch.where(non_zero_bool == True)

    dice_scores = dice_score(sig_pred[non_zero_idx], ground_truth[non_zero_idx])

    return torch.mean(1 - dice_scores), dice_scores
