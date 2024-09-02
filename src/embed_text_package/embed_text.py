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

    def load(self, model_name: str):
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
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        self.which_model = model_name

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
                tqdm_dataloader.set_description(
                    f"Embedding sentences in '{col}' on '{self.model.device}'"
                )

                for sentence in batch[col]:
                    # 1) Get Tokens of sentence
                    tokenized_sentence = self.tokenizer(
                        sentence, add_special_tokens=False, return_tensors="pt"
                    )

                    # 2) Get Embeddings (hiddenstate of last input)
                    # Generate model inputs on same device as self.model
                    # att_mask is vector of ones: Attention on all tokens!

                    tokenized_sentence = {
                        k: v.to(self.model.device)
                        for k, v in tokenized_sentence.items()
                    }
                    # >>>  sequence_length

                    # get embedding via forward function of main self.model.
                    ###########################################################
                    # NOTE: For performance reasons, one could implement
                    # self.model.forward in vectorizedmanner.
                    # If you want to do that, keep padding in mind!
                    ###########################################################
                    sentence_emb = (
                        self.model.forward(**tokenized_sentence)
                        .last_hidden_state[0][-1]
                        .squeeze()
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    # >>> hidden_size

                    # Now just handle list structure.
                    col_emb.append(sentence_emb)
                    # >>> dataset_length x hidden_size

            emb_dict[col] = col_emb
            # >>> num_cols x dataset_length x hidden_size
        emb_dataset = Dataset.from_dict(emb_dict)
        return emb_dataset
