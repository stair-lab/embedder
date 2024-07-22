## Embedding funciton
This python package converts sentences into tokens and passes tokens
through a model to get the sentence embedding. It`s designed to take
multiple batches containing multiple sentences as input.

## How to use:
First, within your environment, install the package.
```bash
pip install git+https://github.com/stair-lab/embedder.git
```
In your script, include the module:
```bash
from embed_text_package.embed_text import Embedder
```

Then you can initialize an embedder, load the model and call it:
> **_NOTE:_** the load() function will load both, the model and embedder.
```bash
model_name = "<HF_repo>/<HF_model>"
embdr = Embedder()
embdr.load(model_name)
emb = embdr.get_embeddings(batches_sentences, model_name)
```
Where `BatchesSentences` is type `list(list(str))` and 
`model_name` is type `str`


## How to test:
First, within your environment, install the package pytest.
```bash
pip install pytest
```
Then, cd to main folder of the package ("embedder") and type:
```bash
pytest
```
