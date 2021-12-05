import pandas as pd
import jsonlines
from tqdm import tqdm
import pickle

from utils.params import BASE_PATH


def data_loader(path, language="spanish", limit=None):
    rv = []
    item_count = 1
    for obj in tqdm(jsonlines.open(path)):
        if obj["language"] != language:
            continue
        rv.append(obj)
        if limit is not None and item_count > limit:
            break
        item_count += 1
    return pd.DataFrame(rv)


def get_embeddings(corpus, force_train=False, **custom_fastsearch_kwargs):
    import gensim
    import pickle

    dump_file_path = "model-data/embeddings.dump"

    if not force_train:
        try:
            with open(dump_file_path, "rb") as f:
                model = pickle.load(f)
                print("Model loaded from file")
        except (FileNotFoundError, EOFError):
            pass
        else:
            return model

    fastsearch_kwargs = {
        "vector_size": 50,
        "window": 3,
        "min_count": 5,
        "epochs": 10,
        "workers": 8,
        "sg": 1,
    }
    fastsearch_kwargs.update(custom_fastsearch_kwargs)

    print(f"Model will be built with params: {fastsearch_kwargs}")

    model = gensim.models.FastText(sentences=corpus, **fastsearch_kwargs)
    with open(dump_file_path, "wb") as f:
        pickle.dump(model, f)

    print("Model built and dumped to file")


def get_false_examples(model, n=0):
    import torch
    from utils.loader import token_title_to_tensor

    with open(BASE_PATH + "spanish_test.pkl", "rb") as fp:
        df = pickle.load(fp)
    with open(BASE_PATH + "selected_top_cats.pkl", "rb") as fp:
        category_map = pickle.load(fp)

    if n > 0:
        df = df.sample(n)

    false_counts = 0
    for row in df[["tokenized_title", "target"]].values:
        x_in = torch.unsqueeze(token_title_to_tensor(row[0]), 0)
        predicted = int(torch.argmax(model(x_in)))
        if predicted != row[1]:
            false_counts += 1
            print(row[0], category_map[predicted], category_map[row[1]])

    print(false_counts / df["target"].count())
