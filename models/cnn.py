import pickle
from pytorch_lightning import callbacks
from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
import torchmetrics
import numpy as np

from utils.loader import MeliDataset, MeliFlattenDataset
from utils.params import BASE_PATH, VECTOR_SIZE

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping


mlf_logger = MLFlowLogger(experiment_name="cnn", tracking_uri="file:./mlruns")

class CNNClassifier(pl.LightningModule):
    def __init__(self,
                 vector_size=VECTOR_SIZE,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=632,
                 dropout=0.5,
                 batch_size=300
                ):
        super().__init__()

        self.batch_size = batch_size
        self.vector_size = vector_size

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.vector_size,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits

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
            MeliDataset("reduced_train_df.pkl"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12
        )
    
    def val_dataloader(self):
        return DataLoader(
            MeliDataset("spanish.validation.pkl"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            MeliDataset("spanish.test.pkl"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12
        )

def set_params(params):
    _params = params


def train(args):
    torch.cuda.set_device(0)
    batch_size = args.batch_size
    vector_size = args.vector_size
    dropout = args.dropout
    epochs = args.epochs

    model = CNNClassifier(vector_size=vector_size, dropout=dropout, batch_size=batch_size)
    early_stopping = EarlyStopping('t n_loss')
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=20, gpus=1, logger=mlf_logger, callbacks=[early_stopping], default_root_dir=BASE_PATH + "mlp_flat/")
    trainer.fit(model)
    trainer.test()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or eval model")
    parser.add_argument("--eval", dest="eval", type=bool, nargs='+', default=False)

    parser.add_argument("--epochs", dest="epochs", type=int, nargs='+', default=4)
    parser.add_argument("--vector_size", dest="vector_size", type=int, nargs='+', default=50)
    parser.add_argument("--batch_size", dest="batch_size", type=int, nargs='+', default=300)
    parser.add_argument("--dropout", dest="dropout", type=float, nargs='+', default=0.5)
    args = parser.parse_args()

    MODEL_PATH = BASE_PATH + "cnn/cnn_model.torch"
    
    if not args.eval:
        model = train(args)
        torch.save(model, MODEL_PATH)
    
    else:
        from utils.loader import token_title_to_tensor
        print("Loading model...")
        model = torch.load(MODEL_PATH)
        model.eval()

        print("Done loading! Exit with 'q'")
        with open(BASE_PATH + "category_map.pkl", "rb") as fp:
            category_map = pickle.load(fp)
        while True:
            title = input("Title: ")
            if title == "q":
                break

            tokens = title.split(" ")

            x_in = torch.unsqueeze(token_title_to_tensor(tokens), 0)
            predicted_class = int(torch.argmax(model(x_in)))
            print(f"Class: {category_map[predicted_class]}")
