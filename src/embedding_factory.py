"""Embedding factory for creating embedding models."""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings


def get_embeddings():
    """Create an embedding function instance based on environment configuration."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbeddings(model=model, api_key=api_key)

    elif provider == "huggingface":
        # Uses sentence-transformers style models
        return HuggingFaceEmbeddings(model_name=model)

    elif provider == "ollama":
        return OllamaEmbeddings(model=model)

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
