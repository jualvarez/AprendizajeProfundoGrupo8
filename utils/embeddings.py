import gensim
import pickle
import pandas as pd
from tqdm import tqdm

from utils.params import BASE_PATH, VECTOR_SIZE


def train_embeddings(corpus, extra_fasttext_kwargs=None):
    if extra_fasttext_kwargs is None:
        extra_fasttext_kwargs = {}

    fasttext_kwargs = {
        "min_count": 5,
        "vector_size": VECTOR_SIZE,
        "workers": 4,
        "window": 8,
        "sg": 1,
    }
    print("Training...")
    fasttext_kwargs.update(extra_fasttext_kwargs)
    model = gensim.models.FastText(sentences=corpus, **fasttext_kwargs)
    print("Done training!")

    with open(BASE_PATH + "embeddings.pkl", "wb") as f:
        pickle.dump(model, f)


def load_embeddings():
    with open(BASE_PATH + "embeddings.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    dataframe_file = BASE_PATH + "cached_dataframe.pkl"
    df = pd.read_pickle(dataframe_file)
    train_embeddings(df["tokenized_title"])
