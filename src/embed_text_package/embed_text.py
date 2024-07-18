"""
embed_text package
"""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModel


def get_embeddings(sentence_batches: list,
                   model: AutoModel, tokenizer: AutoTokenizer):
    """
    Function converts sentences into tokens and passes tokens
    through model to get the sentence embedding. Designed to take
    multiple batches containing multiple sentences as input.
    Here, one sentence is defined in one string (str)

    :param sentence_batches: A list of lists (batches) of sentences
    :type sentence_batches: list(list(string))
    :param model: LLM-style model that transforms tokens & attention mask
        to embeddings
    :type model: tbd
    :param tokenizer: Tokenizer mapping strings to key-values
    :type tokenizer: tbd

    :return: Embeddings of each Sentence
    :rtype: list(list(sentence_emb))
    """

    # for model and tokenizer class, I am not sure.
    # I think model class is LlamaForCausalLM
    # Maybe tokenizer class is "Autotokenizer" (from transformer)

    emb_batches = []

    for batch in tqdm(sentence_batches,
                      ascii=True, desc="Embedding Batches..."):
        batch_emb = []
        for sentence in batch:
            # 1) Get Tokens of sentence

            sentence_tokens = tokenizer(sentence)["input_ids"]

            # 2) Get Embeddings (hiddenstate of last input)
            # Generate model inputs on same device as model
            # att_mask is vector of ones: we want attention on all tokens

            tokens = torch.tensor([sentence_tokens], device=model.device)
            # >>>  sequence_length

            att_mask = torch.tensor([[1] * len(sentence_tokens)],
                                    device=model.device)
            # >>>  sequence_length

            # get embedding by calling forward function of main model.
            ##################################################################
            # TODO: implement model.forward in vectorized manner.
            # NOTE: Check performance difference, take care of padding!
            # model.forward().last_hidden_state has dimension:
            # >>> batch_size x sequence_length x hidden_size
            ##################################################################
            sentence_emb = model.forward(
                input_ids=tokens,
                attention_mask=att_mask).last_hidden_state[0][-1]\
                .squeeze().detach().cpu().tolist()
            # >>> hidden_size

            # Now just handle list structure.
            batch_emb.append(sentence_emb)
            # >>> batch_size x hidden_size

        emb_batches.append(batch_emb)
        # >>> num_batches x batch_size x hidden_size
    return emb_batches
