import pytorch_lightning as pl
import timm
import torch.nn as nn
import torchmetrics
from timm.optim.optim_factory import create_optimizer_v2


def get_timm_model(model_name, pretrained, num_classes, drop_rate, *args, **kwargs):

    model = timm.create_model(
        model_name=model_name, pretrained=pretrained, drop_rate=drop_rate
    )

    if model.num_classes != num_classes:
        model.reset_classifier(num_classes=num_classes)
        model.num_classes = num_classes

    return model


class PLModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = get_timm_model(**kwargs["model"])
        self.optimizer = create_optimizer_v2(self.parameters(), **kwargs["optimizer"])
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.model.num_classes, top_k=1
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log("train_loss", loss)
        self.log("train_acc", accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log("val_loss", loss)
        self.log("val_acc", accuracy)

        return loss
