"""Configuration management for the RAG system."""

import os
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGDomainConfig:
    """Configuration for a single RAG domain."""
    name: str
    display_name: str
    directory_path: str
    file_extensions: str
    description: str
    chunk_size: int = 1000
    chunk_overlap: int = 100

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


@dataclass
class SystemConfig:
    """System-wide configuration."""
    rag_domains: list[RAGDomainConfig]
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 100
    sqlite_path: str = "data/doc_hashes.db"
    chroma_path: str = "data/chroma_db"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.rag_domains:
            raise ValueError("At least one RAG domain must be configured")

        # Check for duplicate domain names
        names = [domain.name for domain in self.rag_domains]
        if len(names) != len(set(names)):
            raise ValueError("RAG domain names must be unique")


def load_config(config_path: str) -> SystemConfig:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    if not config_data or 'rag_domains' not in config_data:
        raise ValueError("Configuration must contain 'rag_domains' section")

    # Get default values from environment or use hardcoded defaults
    default_chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
    default_chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))

    # Parse RAG domains
    rag_domains = []
    for domain_data in config_data['rag_domains']:
        # Apply defaults if not specified
        domain_data.setdefault('chunk_size', default_chunk_size)
        domain_data.setdefault('chunk_overlap', default_chunk_overlap)

        rag_domains.append(RAGDomainConfig(**domain_data))

    # Create system config
    system_config = SystemConfig(
        rag_domains=rag_domains,
        default_chunk_size=default_chunk_size,
        default_chunk_overlap=default_chunk_overlap
    )

    return system_config


def ensure_directories(config: SystemConfig) -> None:
    """Ensure all required directories exist."""
    # Create data directories
    Path(config.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.chroma_path).mkdir(parents=True, exist_ok=True)

    # Create document directories
    for domain in config.rag_domains:
        Path(domain.directory_path).mkdir(parents=True, exist_ok=True)


def get_llm_config() -> dict[str, str]:
    """Get LLM configuration from environment variables."""
    return {
        'provider': os.getenv('DEFAULT_LLM_PROVIDER', 'openai'),
        'model': os.getenv('DEFAULT_MODEL', 'gpt-4o-mini'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
    }
