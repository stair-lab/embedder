"""
embed_text package
"""

import torch


def get_embeddings(sentence_batches: list, model, tokenizer):
    """
    Function converts sentences into tokens and passes tokens
    through model to get the sentence embedding. Designed to take
    multiple batches containing multiple sentences as input.
    Here, one sentence is defined in one string (str)

    :param sentence_batches: A list of lists (batches) of sentences
    :type sentence_batches: list(list(string))
    :param model: LLM-style model consisting of model.model ("main model")
        that transforms tokens & attention mask to embeddings and model.lm_head
        ("linear adapter") that embeddings to next-word prediction
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
    batch_emb = []

    for batch in sentence_batches:
        for sentence in batch:
            # 1) Get Tokens of sentence
            sentence_tokens = tokenizer(sentence)["input_ids"]

            # 2) Get Embeddings (hiddenstate of last input)
            # Generate model inputs on same device as model
            # att_mask is vector of ones: we want attention on all tokens
            tokens = torch.tensor([sentence_tokens], device=model.model.device)
            att_mask = torch.tensor([[1] * len(sentence_tokens)],
                                    device=model.model.device)

            # get embedding by calling forward function of main model.
            sentence_emb = model.model.forward(
                input_ids=tokens,
                attention_mask=att_mask).last_hidden_state[0][-1]\
                .squeeze().detach().cpu().tolist()

            # Now just handle list structure.
            batch_emb.append(sentence_emb)
        emb_batches.append(batch_emb)
    return emb_batches
