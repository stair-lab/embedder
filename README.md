## Embedding funciton
This python package converts sentences into tokens and passes tokens
through a model to get the sentence embedding. It's designed to take
multiple batches containing multiple sentences as input.

## How to use:
First, within your environment, install the package.
```bash
pip install git+https://github.com/stair-lab/embedder.git
```
In your script, include the package:
```bash
from embed_text_package import embed_text
```

Then you can call the get_embeddings() function by
```bash
embed_text.get_embeddings(BatchesSentences,model,tokenizer)
```
Where `BatchesSentences` should be of type `list(list(str))`,`model` should be of type `...tbd...` and `tokenizer` should be of type `...tbd...`