"""
Tests the embed_text module.
"""

import numpy as np
from datasets import Dataset, load_dataset
from embed_text_package.embed_text import Embedder
from torch.utils.data import DataLoader
from transformers import AutoConfig


# TEST DIFFERENT MODEL SIZES ITERATIVELY (small --> large)
# Nested for-loop
def test_workflow():
    """
    test workflow and check dimensions of output.
    """

    dataset_names = ["proteinea/fluorescence", "allenai/reward-bench"]
    ds_split_fluor = "test"
    ds_split_rwben = "raw"

    # Load dataset:
    cols_to_be_embded_fluor = ["primary"]
    cols_to_be_embded_rwben = ["prompt"]
    batch_sizes = [128, 256, 512, 1024]

    # Used model & Tokenizer:
    model_names = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-70b-hf"]
    # no problems with meta-llama/Llama-2-7b-hf (ds: both, bs: 1024)

    for model_name in model_names:
        print(f"model_name = {model_name}\n")
        # Load embedder:
        embdr = Embedder()
        embdr.load(model_name)

        # load model to check dimensions
        config = AutoConfig.from_pretrained(model_name)

        for ds_name in dataset_names:
            if ds_name == "proteinea/fluorescence":
                cols_to_be_embded = cols_to_be_embded_fluor
                ds_split = ds_split_fluor
            elif ds_name == "allenai/reward-bench":
                cols_to_be_embded = cols_to_be_embded_rwben
                ds_split = ds_split_rwben
            else:
                raise ValueError(f"Unknown dataset name: {ds_name}")
            print(f"dataset_name = {ds_name}\n")
            dataset = load_dataset(ds_name)[ds_split]

            for bs in batch_sizes:
                print(f"batch_size = {bs}\n")
                for _ in range(5):
                    rnd = np.random.randint(low=0, high=len(dataset) - bs * 2)
                    sub_ds = Dataset.from_dict(dataset[rnd : rnd + (bs * 2)])

                    # prepare data
                    dataloader = DataLoader(sub_ds.with_format("torch"), bs)

                    # get embeddings
                    emb = embdr.get_embeddings(
                        dataloader, model_name, cols_to_be_embded
                    )

                    # Check dimension:
                    assert len(emb) == len(sub_ds)
                    for col in cols_to_be_embded:
                        assert len(emb[col][0]) == config.hidden_size


if __name__ == "__main__":
    test_workflow()
