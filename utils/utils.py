import pandas as pd
import jsonlines
from tqdm import tqdm


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
        "sg": 1
    }
    fastsearch_kwargs.update(custom_fastsearch_kwargs)
   
    print(f"Model will be built with params: {fastsearch_kwargs}")

    model = gensim.models.FastText(sentences=corpus, **fastsearch_kwargs)
    with open(dump_file_path, "wb") as f:
        pickle.dump(model, f)

    print("Model built and dumped to file")
