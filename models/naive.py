import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from utils.loader import MeliDataset
from utils.embeddings import torch_embeddings

from utils.params import VECTOR_SIZE

_params = {}

class MeliNaive(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        # self.embeddings = torch_embeddings()
        self.hidden1 = nn.Linear(VECTOR_SIZE, 400)
        self.hidden2 = nn.Linear(400, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.layer_out = nn.Linear(1024, 700)
        self.vector_size = VECTOR_SIZE

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(100)
        self.batchnorm2 = nn.BatchNorm1d(100)
        # self.batchnorm3 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        # x = self.embeddings(x)
        # x = torch.mean(x, dim=1)
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)


        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.hidden3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        # x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        # x = torch.sigmoid(self.output(x))
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
            num_workers=12
        )


def set_params(params):
    _params = params

def train():
    torch.cuda.set_device(0)
    model = MeliNaive()
    # from utils.loader import dataloader

    trainer = pl.Trainer(max_epochs=20, progress_bar_refresh_rate=20, gpus=1)
    trainer.fit(model)


if __name__ == "__main__":
    train()