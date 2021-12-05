import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from utils.loader import MeliDataset, PadSequences
from utils.params import VECTOR_SIZE


pad_sequences = PadSequences(pad_value=0, max_length=None, min_length=1)


class MLPClassifier(pl.LightningModule):
    def __init__(
        self,
        n_labels=700,
        hidden_layers=[256, 128],
        dropout=0.3,
        vector_size=VECTOR_SIZE,
    ):
        super().__init__()
        self.hidden_layers = [nn.Linear(vector_size, hidden_layers[0])]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(input_size, output_size))
        self.dropout = dropout
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size

    def forward(self, x):
        print(x.shape)
        x = torch.mean(x, dim=1)
        print(x.shape)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)

        self.log("t n_loss", loss, on_epoch=True)
        # self.log_metrics("accuracy")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(
            MeliDataset(),
            batch_size=4,
            shuffle=True,
            num_workers=12,
            collate_fn=pad_sequences,
        )


def set_params(params):
    _params = params


def train():
    torch.cuda.set_device(0)
    model = MLPClassifier()
    # from utils.loader import dataloader

    trainer = pl.Trainer(max_epochs=20, progress_bar_refresh_rate=20, gpus=1)
    trainer.fit(model)


if __name__ == "__main__":
    train()
