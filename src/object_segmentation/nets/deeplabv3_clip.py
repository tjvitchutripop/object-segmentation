import torch
import torchvision as tv
import pytorch_lightning as pl
from PIL import Image
import torch.nn as nn
import clip

class DeepLabV3_CLIP(pl.LightningModule):
    def __init__(self, image_size, num_classes, hidden_dim, num_heads, num_layers, patch_size, representation_size, mlp_dim, dropout):
        super().__init__()

        # Load the DeepLabV3 model
        self.deeplabv3 = tv.models.segmentation.deeplabv3_resnet101(
            image_size=image_size,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
            representation_size=representation_size,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.model, _ = clip.load("ViT-B/32", device="cuda")

        # self.linear = nn.Linear(512, 256)

        # You can customize layers here, for example:
        # Modify the first layer
        self.deeplabv3.backbone.conv1 = nn.Conv2d(
            in_channels=515,  # Modify this to match the number of input channels
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    def forward(self, img, text):
        text_tokens = clip.tokenize(text).to("cuda")
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        text_features = text_features.unsqueeze(1).unsqueeze(1).repeat(1, img.shape[2], img.shape[3], 1)
        text_features = text_features.permute(0, 3, 1, 2)
        concatenated_image = torch.cat((img, text_features), dim=1)
        return self.deeplabv3(concatenated_image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer