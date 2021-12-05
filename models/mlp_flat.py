import pickle
from pytorch_lightning import callbacks
from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
import torchmetrics

from utils.loader import MeliFlattenDataset
from utils.params import BASE_PATH, VECTOR_SIZE

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping


mlf_logger = MLFlowLogger(experiment_name="mlp_flat", tracking_uri="file:./mlruns")


class MLPClassifier(pl.LightningModule):
    def __init__(
        self,
        n_labels=700,
        hidden_layers=[1024, 2048, 2048, 1024],
        dropout=0.3,
        vector_size=VECTOR_SIZE,
    ):
        super().__init__()
        self.hidden_layers = [nn.Linear(vector_size * 50, hidden_layers[0])]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(input_size, output_size))
        self.dropout = dropout
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        preds = self(x)

        loss = F.cross_entropy(self(x), y)

        self.log("t n_loss", loss, on_epoch=True)
        self.log("step_accuracy", self.accuracy(preds, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy(preds, y))
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def train_dataloader(self):
        return DataLoader(
            MeliFlattenDataset("reduced_train_df.pkl"),
            batch_size=300,
            shuffle=True,
            num_workers=12,
        )

    def val_dataloader(self):
        return DataLoader(
            MeliFlattenDataset("spanish.validation.pkl"),
            batch_size=300,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            MeliFlattenDataset("spanish.test.pkl"),
            batch_size=300,
            shuffle=False,
            num_workers=12,
        )


def set_params(params):
    _params = params


def train():
    torch.cuda.set_device(0)
    model = MLPClassifier()
    early_stopping = EarlyStopping("t n_loss")
    trainer = pl.Trainer(
        max_epochs=1,
        progress_bar_refresh_rate=20,
        gpus=1,
        logger=mlf_logger,
        callbacks=[early_stopping],
        default_root_dir=BASE_PATH + "mlp_flat/",
    )
    trainer.fit(model)
    trainer.test()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or eval model")
    parser.add_argument("--eval", dest="eval", type=bool, nargs="+", default=False)
    args = parser.parse_args()

    MODEL_PATH = BASE_PATH + "mlp_flat/mlp_flat_model.torch"

    if not eval:
        model = train()
        torch.save(model, MODEL_PATH)

    else:
        from utils.loader import token_title_to_tensor

        print("Loading model...")
        model = torch.load(MODEL_PATH)
        print("Done loading! Exit with 'q'")
        with open(BASE_PATH + "category_map.pkl", "rb") as fp:
            category_map = pickle.load(fp)
        while True:
            title = input("Title: ")
            if title == "q":
                break

            tokens = title.split(" ")

            x_in = torch.flatten(token_title_to_tensor(tokens))
            predicted_class = int(torch.argmax(model(x_in)))
            print(f"Class: {category_map[predicted_class]}")
