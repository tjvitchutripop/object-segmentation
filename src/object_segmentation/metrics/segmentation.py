import pandas as pd
from torchmetrics import JaccardIndex
import torch

def get_metrics(preds, targets):
    preds = preds["out"].squeeze().to(torch.float32)
    jaccard = JaccardIndex(task="binary", num_classes=1)
    average_iou = jaccard(preds, targets)

    return {
        "average_iou": average_iou,
    }
