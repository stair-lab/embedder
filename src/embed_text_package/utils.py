from vllm import ModelRegistry
from vllm.model_executor.models.registry import _EMBEDDING_MODELS

from .models import DeepseekEmbeddingModel, LlamaEmbeddingModel

global _EMBEDDING_MODELS


def register_model():
    ModelRegistry.register_model("LlamaEmbModel", LlamaEmbeddingModel)
    _EMBEDDING_MODELS["LlamaEmbModel"] = LlamaEmbeddingModel

    ModelRegistry.register_model("DeepseekEmbModel", DeepseekEmbeddingModel)
    _EMBEDDING_MODELS["DeepseekEmbModel"] = DeepseekEmbeddingModel
