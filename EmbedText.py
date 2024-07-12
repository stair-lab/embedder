import torch


def get_embeddings(SentenceBatches:list(list(string)), model, tokenizer):
    #for model and tokenizer class, I am not sure.
    # I think model class is LlamaForCausalLM
    # Maybe tokenizer class is "Autotokenizer" (from transformer)
    """
    Function converts sentences into tokens and passes tokens 
    through model to get the sentence embedding. Designed to take
    multiple batches containing multiple sentences as input.
    Here, one sentence is defined in one string (str)

    :param SentenceBatches: A list of lists (batches) of sentences
    :type SentenceBatches: list(list(string))
    :param model: LLM-style model that transforms tokens to embeddings
    :type model: 
    :param tokenizer: Tokenizer mapping strings to key-values
    :type tokenizer: 

    :return: Embeddings of each Sentence
    :rtype: list(list(SentenceEmbedding))
    """

    EmbeddingBatches = []
    BatchEmbedding = []

    for batch in SentenceBatches:
        for sentence in batch:
            ## Get Tokens of sentence
            SentenceTokens = tokenizer(sentence)["input_ids"]

            ## Get Embeddings (hiddenstate of last input)
            # make sure, model inputs are on same device as model
            # AttMask is vector of ones: we want attention on all tokens
            InpIds=torch.tensor([SentenceTokens],device=model.model.device)
            AttMask=torch.tensor([[1] * len(SentenceTokens)],\
                device=model.model.device)

            # get embedding by calling forward function of main model. 
            # Model consists of model.model("main model") 
            #   and model.lm_head ("linear adapter")
            SentenceEmbedding = model.model.forward(\
                    input_ids=InpIds,\
                    attention_mask=AttMask).last_hidden_state[0][-1]\
                    .squeeze().detach().cpu().tolist()

            # Now just handle list structure.
            BatchEmbedding.append(SentenceEmbedding)
        EmbeddingBatches.append(BatchEmbedding)
    return EmbeddingBatches
