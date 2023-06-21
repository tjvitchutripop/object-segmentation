from typing import Any

import lightning as L
import pandas as pd
import torch
import torchmetrics.functional.classification as tfc
import torchvision as tv
import wandb

from python_ml_project_template.datasets.cifar10 import CIFAR10DataModule
from python_ml_project_template.utils.script_utils import PROJECT_ROOT, match_fn


class ClassifierEvalModule(L.LightningModule):
    def __init__(self, network) -> None:
        super().__init__()
        self.network = network

    def forward(self, x):
        self.network(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        imgs, labels = batch
        preds = self.network(imgs)
        return {"preds": preds, "labels": labels}


@torch.no_grad()
def main():
    # Global seed for reproducibility.
    L.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")
    # run_id = "pjf0nfg6"
    run_id = "0v36p8tn"
    checkpoint_reference = f"r-pad/lightning-hydra-template/model-{run_id}:v0"

    # download checkpoint locally (if not already cached)
    run = wandb.init(
        entity="r-pad",
        project="lightning-hydra-template",
        job_type="eval",
        group=f"experiment-{run_id}",
        config={"eval config": "wat"},
    )

    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

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

    artifact = run.use_artifact(checkpoint_reference, type="model")
    # artifact_dir = artifact.download()
    ckpt_file = artifact.get_path("model.ckpt").download()

    ckpt = torch.load(ckpt_file)

    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )

    model = ClassifierEvalModule(network)

    root = "./data"
    datamodule = CIFAR10DataModule(root, batch_size=128, num_workers=4)
    # Gotta call this in order to establish the dataloaders.
    datamodule.setup("predict")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=1,
        logger=False,
    )

    train_preds, val_preds, test_preds = trainer.predict(
        model,
        dataloaders=[
            *datamodule.val_dataloader(),  # There are two different loaders (train_val and val).
            datamodule.test_dataloader(),
        ],
    )

    # Each of these is a list of dictionaries, where each dictionary is the output of the predict_step method.
    # We can use the `preds` and `labels` keys to calculate metrics.
    # For example, we can calculate the accuracy like so:

    for pred_list, name in [
        (train_preds, "train"),
        (val_preds, "val"),
        (test_preds, "test"),
    ]:
        pass

        all_preds = torch.cat([x["preds"].cpu() for x in pred_list])
        all_labels = torch.cat([x["labels"].cpu() for x in pred_list])
        global_acc = (
            tfc.multiclass_accuracy(
                all_preds,
                all_labels,
                num_classes=10,
                average="micro",
            )
            .numpy()
            .item()
        )

        macro_acc = (
            tfc.multiclass_accuracy(
                all_preds,
                all_labels,
                num_classes=10,
                average="micro",
            )
            .numpy()
            .item()
        )

        # We also want to log per-label accuracies.
        acc_per_label = tfc.multiclass_accuracy(
            all_preds,
            all_labels,
            num_classes=10,
            average="none",
        ).numpy()

        # Create a dataframe with the per-label accuracies, as well as the global and macro accuracies.
        # The columns of the table should be the labels, and there should only be a single row.
        acc_df = pd.DataFrame(acc_per_label[None], columns=[str(i) for i in range(10)])

        # Log the dataframe to wandb.
        table = wandb.Table(dataframe=acc_df)
        run.log({f"{name}_accuracy_table": table})

        run.summary[f"{name}_true_accuracy"] = global_acc
        run.summary[f"{name}_class_balanced_accuracy"] = macro_acc


if __name__ == "__main__":
    main()
