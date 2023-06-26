import pandas as pd
from torchmetrics import JaccardIndex

def get_metrics(preds, targets):

    jaccard = JaccardIndex(num_classes=2)
    average_iou = jaccard(preds, targets)

    return {
        "Average IoU": average_iou,
    }
