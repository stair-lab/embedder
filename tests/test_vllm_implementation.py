import gc

import torch
from datasets import Dataset
from embed_text_package.embed_text import Embedder
from embed_text_package.embed_text_v2 import Embedder as EmbedderV2

ds = Dataset.from_dict({"text": ["hello world"]})
# {'input_ids': tensor([[128000, 15339,  1917]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1]], device='cuda:0')}

# Load first implementation
embdr = Embedder()
embdr.load(
    "meta-llama/Meta-Llama-3-8B",
)
embdr.model.to(dtype=torch.float16)

# Run first implementation
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
emb = embdr.get_embeddings(dataloader, "meta-llama/Meta-Llama-3-8B", ["text"])
embs = torch.tensor(emb["text"])

# Free memory and load second implementation
del embdr
gc.collect()
torch.cuda.empty_cache()

# Load second implementation
embdr = EmbedderV2()
embdr.load(
    "meta-llama/Meta-Llama-3-8B",
    dtype=torch.float16,
)

# Run second implementation
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
emb = embdr.get_embeddings(dataloader, "meta-llama/Meta-Llama-3-8B", ["text"])
embsv2 = torch.tensor(emb["text"])

assert torch.abs(embs - embsv2).mean() < 3e-3
