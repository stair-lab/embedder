"""
embed_text package
"""

import gc

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import register_model

register_model()

from vllm import LLM


class Embedder:
    """
    Instances of this class can embed sentences to embeddings.
    """

    def __init__(self):
        """
        Initialize class object.
        """
        self.model = None
        self.which_model = None

    def load(self, model_name: str, *arg, **kwargs):
        """
        Loads class variables: model and tokenizer.

        :param model_name: HF model name (used for model and tokenizer)
                           format: "hf_repo/hf_model"
        :type model_name: str
        :param self.model: LLM-style model that transforms tokens & attention
        mask to embeddings
        :type self.model: AutoModel
        :param self.tokenizer: Tokenizer mapping strings to key-values
        :type self.tokenizer: AutoTokenizer
        :param which_model: Variable storing the name of the loaded model
        :type which_model: str
        """
        self.model = LLM(model=model_name, *arg, **kwargs)
        self.which_model = model_name

    def unload(self):
        """
        Unloads class variables: model and tokenizer
        """
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_embeddings(self, dataloader: DataLoader, model_name: str, cols: list):
        """
        Function converts sentences to sentence embeddings. Designed to take
        dataloader format as input. Dataset of dataloader should contain
        sentences in string format.

        :param dataloader: Dataloader object of pytorch
        :type dataloader: DataLoader
        :param model_name: HF model name (used for model and tokenizer)
                           format: "hf_repo/hf_model"
        :type model_name: str
        :param cols: list of column names to be embedded
        :type cols: list


        :return: Dataset with columns cols and embeddings of sentences
        :rtype: Dataset
        """
        assert (
            model_name == self.which_model
        ), f"Model '{model_name}' is not preloaded. Loaded model is \
            '{self.which_model}'. Load the correct model by calling the load \
            function."

        emb_dict = {}

        for col in cols:
            col_emb = []
            tqdm_dataloader = tqdm(dataloader)
            for batch in tqdm_dataloader:
                encoded = self.model.encode(batch[col])
                col_emb.extend([x.outputs.embedding for x in encoded])

            emb_dict[col] = col_emb
            # >>> num_cols x dataset_length x hidden_size
        emb_dataset = Dataset.from_dict(emb_dict)
        return emb_dataset
