import torch


def get_embeddings(SentenceBatches, model, tokenizer):
    # Input:    - SentenceBathes: A list of batches of sentences list(list(str))
    #             one sentence is one string str.
    #           - model: as defined in llmtuner.model
    #           - tokenizer: as defined in llmtuner.model
    # Output:   - EmbeddingBatches: Embeddings of each Sentence
    #             according to SentenceBatches structure: list(list(embeddings))
    # Explanation: Function converts sentences into tokens and gets their embedding.
    #             Can be called with multiple batches containing multiple sentences.
    
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
            AttMask=torch.tensor([[1] * len(SentenceTokens)],device=model.model.device)

            # get the embedding by calling the forward function of the main model. 
            # Model consists of model.model("main model") 
            #               and model.lm_head ("linear adapter")
            SentenceEmbedding = model.model.forward(\
                    input_ids=InpIds,\
                    attention_mask=AttMask).last_hidden_state[0][-1]\
                    .squeeze().detach().cpu().tolist()

            # Now just handle list structure.
            BatchEmbedding.append(SentenceEmbedding)
        EmbeddingBatches.append(BatchEmbedding)
    return EmbeddingBatches
