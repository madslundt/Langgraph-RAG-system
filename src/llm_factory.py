"""LLM factory for creating different language model instances."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Create an LLM instance based on provider and model.

    Args:
        provider: LLM provider ('openai', 'anthropic')
        model: Model name
        **kwargs: Additional arguments passed to the LLM constructor

    Returns:
        LLM instance

    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    # Use defaults from environment if not provided
    if provider is None:
        provider = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')

    # Create LLM instance based on provider
    if provider.lower() == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.1),
            **{k: v for k, v in kwargs.items() if k != 'temperature'}
        )

    elif provider.lower() == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")

        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.1),
            **{k: v for k, v in kwargs.items() if k != 'temperature'}
        )

    elif provider.lower() == 'ollama':
        return ChatOllama(
            model=model,
            temperature=kwargs.get('temperature', 0.1),
            **{k: v for k, v in kwargs.items() if k != 'temperature'}
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def validate_llm_config() -> dict:
    """Validate LLM configuration and return status.

    Returns:
        Dictionary with validation results
    """
    results = {
        'openai': {
            'api_key_present': bool(os.getenv('OPENAI_API_KEY')),
            'available': False
        },
        'anthropic': {
            'api_key_present': bool(os.getenv('ANTHROPIC_API_KEY')),
            'available': False
        },
        'ollama': {
            'available': False
        }
    }

    # Test OpenAI
    if results['openai']['api_key_present']:
        try:
            llm = get_llm('openai', 'gpt-3.5-turbo')
            # Simple test call
            response = llm.invoke("Hello")
            results['openai']['available'] = True
        except Exception as e:
            results['openai']['error'] = str(e)

    # Test Anthropic
    if results['anthropic']['api_key_present']:
        try:
            llm = get_llm('anthropic', 'claude-3-haiku-20240307')
            # Simple test call
            response = llm.invoke("Hello")
            results['anthropic']['available'] = True
        except Exception as e:
            results['anthropic']['error'] = str(e)

    return results
