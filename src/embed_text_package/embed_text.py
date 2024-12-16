"""
embed_text package
"""

import gc

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class Embedder:
    """
    Instances of this class can embed sentences to embeddings.
    """

    def __init__(self):
        """
        Initialize class object.
        """
        self.tokenizer = None
        self.model = None
        self.which_model = None

    def load(self, model_name: str, special_tokens=True):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.which_model = model_name
        self.special_tokens = special_tokens

    def unload(self):
        """
        Unloads class variables: model and tokenizer
        """
        del self.tokenizer
        del self.model
        self.tokenizer = None
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
                model_inputs = self.tokenizer(
                    batch[col],
                    add_special_tokens=self.special_tokens,
                    return_tensors="pt",
                    padding=True,
                )
                model_inputs = {
                    k: v.to(self.model.device) for k, v in model_inputs.items()
                }
                embeds = self.model(**model_inputs)

                last_idxs = []
                for i in range(embeds.last_hidden_state.size(0)):
                    if self.tokenizer.pad_token_id is None:
                        end_index = -1
                    else:
                        end_indexes = (
                            model_inputs["input_ids"][i] != self.tokenizer.pad_token_id
                        ).nonzero()
                        end_index = end_indexes[-1].item() if len(end_indexes) else 0

                    last_idxs.append(end_index)

                embed_last_token = (
                    embeds.last_hidden_state[list(range(len(last_idxs))), last_idxs]
                    .cpu()
                    .tolist()
                )
                col_emb.extend(embed_last_token)

            emb_dict[col] = col_emb
            # >>> num_cols x dataset_length x hidden_size
        emb_dataset = Dataset.from_dict(emb_dict)
        return emb_dataset
