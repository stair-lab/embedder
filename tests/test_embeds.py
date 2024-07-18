
from embed_text_package import embed_text
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel

# Load dataset:
dataset = load_dataset("proteinea/fluorescence")["test"]

# Load (pre-trained) Tokenizer (according to model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf").to('cuda')

def test_workflow():

    # create batch-structure:
    batch_size = 64
    batches_sentences = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batches_sentences.append(dataset['primary'][i:i+batch_size])

    # Time-reasons: only do first and last batch for now
    batches_sentences = [batches_sentences[0], batches_sentences[-1]]
    # get embeddings
    emb = embed_text.get_embeddings(batches_sentences, model, tokenizer)


    # Check dimension:
    assert(len(batches_sentences[0]) == batch_size) #batch
    # missing: check hidden_size
    # missing: check num of batches


if __name__ == "__main__":
    test_workflow()