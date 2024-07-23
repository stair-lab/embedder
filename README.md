## Embedding funciton
This python package converts sentences into tokens and passes tokens
through a model to get the sentence embedding. Designed to take dataloader
format as input.

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
emb = embdr.get_embeddings(dataloader, MODEL_NAME, cols_to_be_embded)
```
Where `dataloader` is type `Dataloader`, 
`model_name` is type `str`.
`cols_to_be_embded` is type `list` and should contain the names of the columns
of the dataloader dataset which shall be embedded.


## How to test:
First, within your environment, install the package pytest.
```bash
pip install pytest
```
Then, cd to main folder of the package ("embedder") and type:
```bash
pytest
```
