import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torchmetrics
from timm.optim.optim_factory import create_optimizer_v2


def get_timm_model(model_name, pretrained, num_classes, drop_rate, checkpoint_path):

    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        checkpoint_path=checkpoint_path,
    )

    if model.num_classes != num_classes:
        model.reset_classifier(num_classes=num_classes)
        model.num_classes = num_classes

    return model


class PLModel(pl.LightningModule):
    """
    Validation Step x2
    Validation Epoch End

    Loop {
    Training Step
    Validation Step
    Validation Epoch End
    Training Epoch End
    }
    """

    def __init__(self, model_config, optimizer_config):
        super().__init__()
        self.model = get_timm_model(**model_config)
        self.optimizer = create_optimizer_v2(self.parameters(), **optimizer_config)
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.model.num_classes, top_k=1
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.model.num_classes, top_k=1
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)
        accuracy = self.train_accuracy(y_pred, y)

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", accuracy, on_epoch=True, on_step=False)

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.train_accuracy.reset()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)
        accuracy = self.val_accuracy(y_pred, y)

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_acc", accuracy, on_epoch=True, on_step=False)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.val_accuracy.reset()
