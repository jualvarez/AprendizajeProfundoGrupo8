from typing import Optional
import jsonlines
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch
from tqdm import tqdm
import numpy as np
from sklearn.utils import compute_class_weight

from utils.params import BASE_PATH, NUM_CLASSES, VECTOR_SIZE

import pickle


def dataset_loader(path, output="cached_dataframe.pkl", limit=None, categories=False):

    row_count = 1
    rows = []
    if categories:
        with open(BASE_PATH + "selected_top_cats.pkl", "rb") as fp:
            cats = pickle.load(fp)
    for line in tqdm(jsonlines.open(path)):
        if categories and line["category"] not in cats:
            continue
        if not len(line["tokenized_title"]):
            continue

        rows.append(line)
        if limit is not None and row_count > limit:
            break
        row_count += 1

    dataframe = pd.DataFrame(rows)
    dataframe.to_pickle(BASE_PATH + output)
    return dataframe


def get_categories_weight(train_file="reduced_train_df.pkl"):
    with open(BASE_PATH + train_file, "rb") as fp:
        df = pickle.load(fp)

    classes = np.unique(df["target"])
    class_weight = compute_class_weight("balanced", classes=classes, y=df["target"])
    weight_complete = [
        class_weight[np.where(classes == i)[0][0]] if i in classes else 1
        for i in range(NUM_CLASSES)
    ]
    tensor = torch.tensor(weight_complete)
    with open(BASE_PATH + "class_weights.pkl", "wb") as fp:
        pickle.dump(tensor, fp)
    return tensor


class TokensToTensor:
    embeddings = None

    def __init__(self) -> None:
        print("Loading embeddings")
        from utils.embeddings import load_embeddings

        self.embedding = load_embeddings()
        print("Embeddings loaded")

    def __call__(self, tokens):
        tokenized_title = tokens[:50]
        data_array = np.array([self.embedding.wv[w] for w in tokenized_title])
        tensor = torch.from_numpy(data_array)
        data = F.pad(tensor, (0, 0, 0, 50 - tensor.size(0)))
        return data


token_title_to_tensor = TokensToTensor()


class MeliDataset(Dataset):
    def __init__(self, dataset_file):
        try:
            self.dataset = pd.read_pickle(BASE_PATH + dataset_file)
        except FileNotFoundError:
            self.dataset = pd.DataFrame(["sadf"])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        data = token_title_to_tensor(item["tokenized_title"])
        target = item["target"]
        # data = self.embedding.wv[item["tokenized_title"][0]]
        return data, target


class MeliFlattenDataset(Dataset):
    def __init__(self, dataset_file):
        try:
            self.dataset = pd.read_pickle(BASE_PATH + dataset_file)
        except FileNotFoundError:
            self.dataset = pd.DataFrame(["sadf"])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        data = token_title_to_tensor(item["tokenized_title"])
        target = item["target"]
        # data = self.embedding.wv[item["tokenized_title"][0]]
        return torch.flatten(data), target


class PadSequences:
    def __init__(self, pad_value=0, max_length=None, min_length=1):
        assert max_length is None or min_length <= max_length
        self.pad_value = pad_value
        self.max_length = max_length
        self.min_length = min_length

    def __call__(self, items):
        data = [item[0] for item in items]
        target = [item[1] for item in items]
        seq_lengths = [len(d) for d in data]

        if self.max_length:
            max_length = self.max_length
            seq_lengths = [min(self.max_length, l) for l in seq_lengths]
        else:
            max_length = max(self.min_length, max(seq_lengths))

        data = [
            d[:l] + [self.pad_value] * (max_length - l)
            for d, l in zip(data, seq_lengths)
        ]

        print(torch.LongTensor(data), torch.LongTensor(target))
        return torch.LongTensor(data), torch.LongTensor(target)


class MeliDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = BASE_PATH, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # self.mnist_test = MNIST(self.data_dir, train=False)
        # self.mnist_predict = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.train_data = MeliDataset()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=12)

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict, batch_size=self.batch_size, num_workers=12
        )

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load dataset and cache in disk")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("limit", type=int, nargs="?", default=None)
    parser.add_argument(
        "--categories", dest="categories", type=bool, nargs="?", default=False
    )
    args = parser.parse_args()
    print("Loading data...")
    dataset_loader(
        BASE_PATH + args.input,
        output=args.output,
        limit=args.limit,
        categories=args.categories,
    )
