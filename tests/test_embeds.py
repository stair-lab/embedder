"""
Tests the embed_text module.
"""

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoModel
from embed_text_package.embed_text import Embedder

# Load dataset:
dataset = load_dataset("proteinea/fluorescence")["test"]
dataset = Dataset.from_dict(dataset[0:1000])  # only first 1000 sentences!
cols_to_be_embded = ["primary"]
BATCH_SIZE = 64

# Used model & Tokenizer:
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Load embedder:
embdr = Embedder()
embdr.load(MODEL_NAME)


def test_workflow():
    """
    test workflow and check dimensions of output.
    """
    # prepare data
    dataloader = DataLoader(dataset.with_format("torch"), BATCH_SIZE)

    # get embeddings
    emb = embdr.get_embeddings(dataloader, MODEL_NAME, cols_to_be_embded)

    # load model to check dimensions
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Check dimension:
    assert len(emb) == len(dataset)
    for col in cols_to_be_embded:
        assert len(emb[col][0]) == model.config.hidden_size


if __name__ == "__main__":
    test_workflow()
