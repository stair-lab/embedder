"""
Tests the embed_text module.
"""
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel
from embed_text_package.embed_text import Embedder

# Load dataset:
dataset = load_dataset("proteinea/fluorescence")["test"]

# Used model & Tokenizer:
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Load embedder:
embdr = Embedder()
embdr.load(MODEL_NAME)


def test_workflow():
    """
    test workflow and check dimensions of output.
    """

    # create batch-structure:
    batch_size = 64
    batches_sentences = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batches_sentences.append(dataset["primary"][i : i + batch_size])

    # Time-reasons: only do first and last batch for now
    batches_sentences = [batches_sentences[0], batches_sentences[-1]]
    # get embeddings
    emb = embdr.get_embeddings(batches_sentences, MODEL_NAME)

    # Check dimension:
    assert len(emb[0]) == batch_size
    # load model to check dimensions
    model = AutoModel.from_pretrained(MODEL_NAME)
    assert len(emb[0][0]) == model.config.hidden_size
    # Time-reasons: only do first and last batch for now
    # if len(dataset)%batch_size != 0:
    #    assert (len(dataset)//batch_size +1 == len(emb))
    # else:
    #     assert (len(dataset)//batch_size == len(emb))


if __name__ == "__main__":
    test_workflow()
