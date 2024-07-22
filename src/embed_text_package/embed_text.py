"""
embed_text package
"""
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModel


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

    def get_embeddings(self, sentence_batches: list, model_name: str):
        """
        Function converts sentences into tokens and passes tokens
        through model to get the sentence embedding. Designed to take
        multiple batches containing multiple sentences as input.
        Here, one sentence is defined in one string (str)

        :param sentence_batches: A list of lists (batches) of sentences
        :type sentence_batches: list(list(string))
        :param model_name: HF model name (used for model and tokenizer)
                           format: "hf_repo/hf_model"
        :type model_name: str


        :return: Embeddings of each Sentence
        :rtype: list(list(sentence_emb))
        """
        assert (
            model_name == self.which_model
        ), f"Model '{model_name}' is not preloaded. Loaded model is \
            '{self.which_model}'. Load the correct model by calling the load \
            function."

        emb_batches = []

        for batch in tqdm(
            sentence_batches, ascii=True, desc="Embedding Batches..."
        ):
            batch_emb = []
            for sentence in batch:
                # 1) Get Tokens of sentence

                sentence_tokens = self.tokenizer(sentence)["input_ids"]

                # 2) Get Embeddings (hiddenstate of last input)
                # Generate model inputs on same device as self.model
                # att_mask is vector of ones: we want attention on all tokens

                tokens = torch.tensor(
                    [sentence_tokens], device=self.model.device
                )
                # >>>  sequence_length

                att_mask = torch.tensor(
                    [[1] * len(sentence_tokens)], device=self.model.device
                )
                # >>>  sequence_length

                # get embedding by calling forward function of main self.model.
                ###############################################################
                # NOTE: One could implement self.model.forward in vectorized
                # manner. Check performance difference, take care of padding!
                # self.model.forward().last_hidden_state has dimension:
                # >>> batch_size x sequence_length x hidden_size
                ###############################################################
                sentence_emb = (
                    self.model.forward(
                        input_ids=tokens, attention_mask=att_mask
                    )
                    .last_hidden_state[0][-1]
                    .squeeze()
                    .detach()
                    .cpu()
                    .tolist()
                )
                # >>> hidden_size

                # Now just handle list structure.
                batch_emb.append(sentence_emb)
                # >>> batch_size x hidden_size

            emb_batches.append(batch_emb)
            # >>> num_batches x batch_size x hidden_size
        return emb_batches
