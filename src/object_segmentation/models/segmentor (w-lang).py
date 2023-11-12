from typing import Any

import lightning as L
import torch.nn.functional as F
from torch import optim
from object_segmentation.metrics.segmentation import get_metrics
import torch
from torchmetrics import JaccardIndex



class SegmentorTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg) -> None:
        super().__init__()
        self.network = network
        self.lr = training_cfg.lr

    def forward(self, x):
        self.network(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[100, 150], gamma=0.1
        # )
        return [optimizer]#, [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, target, text = batch
        preds = self.network(imgs, text)
        preds = preds["out"].squeeze()
        target = target.squeeze().to(torch.float32)
        loss = F.binary_cross_entropy_with_logits(preds, target)
        istrain = mode == "train"
        preds = preds.to(torch.float32)
        jaccard = JaccardIndex(task="binary", num_classes=1).to(torch.device('cuda'))
        average_iou = jaccard(preds, target)
        self.log("%s_loss" % mode, loss, prog_bar=istrain, add_dataloader_idx=False)
        self.log("%s_average_iou" % mode, average_iou, add_dataloader_idx=False)
        return {"loss": loss, "preds": preds, "average_iou":average_iou}

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            mode = "train_val"
        else:
            mode = "val"
        return self._calculate_loss(batch, mode=mode)

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")


class SegmentorInferenceModule(L.LightningModule):
    def __init__(self, network) -> None:
        super().__init__()
        self.network = network

    def forward(self, x):
        self.network(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        imgs, target, text = batch
        preds = self.network(imgs, text)
        return {"imgs": imgs, "preds": preds, "target": target, "text":text}
