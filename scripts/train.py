import lightning as L
import torch
import torch.nn.functional as F
import torchvision as tv
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import optim

from python_ml_project_template.datasets.cifar10 import CIFAR10DataModule
from python_ml_project_template.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    match_fn,
)

# TODOs:
# [x] Switch to CIFAR10
# [x] Add wandb, including saving the checkpoint, logging an image, and saving the codebase state.
#     - [x] Add a callback to save the model to wandb.
#     - [x] Add a callback to save the codebase to wandb.
#     - [x] Add an image logging example.
# [x] Add a DataModule
# [x] Add an eval script which loads from wandb, and outputs a table.
# [ ] Fix grouping stuff...
# [ ] Add hydra configs
# [ ] Align the checkpoints and log files


class ClassifierTrainingModule(L.LightningModule):
    def __init__(self, network, lr: float) -> None:
        super().__init__()
        self.network = network
        self.lr = lr

    def forward(self, x):
        self.network(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.network(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        istrain = mode == "train"
        self.log("%s_loss" % mode, loss, prog_bar=istrain, add_dataloader_idx=False)
        self.log("%s_acc" % mode, acc, add_dataloader_idx=False)
        return {"loss": loss, "acc": acc, "preds": preds}

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


def main():
    # Global seed for reproducibility.
    L.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    root = "./data"

    network = tv.models.VisionTransformer(
        image_size=32,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        patch_size=4,
        num_classes=10,
        representation_size=256,
        mlp_dim=2048,
        dropout=0.2,
    )
    model = ClassifierTrainingModule(network, lr=3e-4)

    datamodule = CIFAR10DataModule(root, batch_size=128, num_workers=4)

    save_dir = "./wandb"
    checkpoint_dir = "./checkpoints"

    # Create a new wandb run.
    id = wandb.util.generate_id()
    group = "experiment-" + id

    logger = WandbLogger(
        project="lightning-hydra-template",
        entity="r-pad",
        log_model=True,  # Only log the last checkpoint to wandb, and only the LAST model checkpoint.
        save_dir=save_dir,
        config={"testit": "wat"},
        job_type="train",
        group=group,
        id=id,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=1,
        logger=logger,
        callbacks=[
            LogPredictionSamplesCallback(logger),
            # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
            # It saves everything, and you can load by referencing last.ckpt.
            ModelCheckpoint(
                checkpoint_dir,
                filename="{epoch}-{step}",
                monitor="step",
                mode="max",
                save_weights_only=False,
                save_last=True,
            ),
            # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
            ModelCheckpoint(
                checkpoint_dir,
                filename="{epoch}-{step}-{val_loss:.2f}-weights-only",
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
            ),
        ],
    )

    # Log the code used to train the model. Make sure not to log too much, because it will be too big.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    # Run training.
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
