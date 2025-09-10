# LangGraph RAG System

A configurable RAG (Retrieval-Augmented Generation) system built with LangGraph and LangChain that supports multiple document domains, efficient change detection, and interactive chat with source citations.

## Features

* **Multi-domain RAG**: Configure multiple RAG domains (e.g., User Guide, Infrastructure, API docs)
* **Efficient ingestion**: File-level and chunk-level hash tracking to avoid unnecessary re-processing
* **Smart chunking**: Configurable chunk size and overlap per domain
* **Source citations**: Human-readable source references in chat responses
* **LLM flexibility**: Easy switching between different LLMs (OpenAI, Anthropic, etc.)
* **Interactive chat**: Persistent conversation memory within sessions

## Quick Start

### 1\. Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
git clone https://github.com/madslundt/Langgraph-RAG-system
cd langgraph-rag-system

# Install dependencies
uv sync
```

### 2\. Environment Setup

Create a `.env` file based on [.env.example](.env.example):

```bash
# Required: Choose your LLM provider
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Customize default settings
DEFAULT_LLM_PROVIDER=openai  # or anthropic
DEFAULT_MODEL=gpt-4o-mini    # or claude-3-haiku-20240307
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

### 3\. Configuration

Create your RAG domains configuration in `config/rag_config.yaml`:

```yaml
rag_domains:
  - name: "user_guide"
    display_name: "User guide"
    directory_path: "docs/user_guide"
    description: "This tool retrieves answers from the User Guide documentation."
    chunk_size: 1000
    chunk_overlap: 100

  - name: "infrastructure"
    display_name: "Infrastructure"
    directory_path: "docs/infra"
    description: "This tool retrieves answers from Infrastructure setup and operations docs."
    chunk_size: 1500
    chunk_overlap: 150
```

### 4\. Add Your Documents

```bash
mkdir -p docs/user_guide docs/infra
# Add your PDF and Markdown files to the respective directories
```

### 5\. Usage

```bash
# Populate/update RAG data
uv run python main.py populate --config config/rag_config.yaml

# Clean stale documents (optional)
uv run python main.py clean --config config/rag_config.yaml

# Start interactive chat
uv run python main.py chat --config config/rag_config.yaml

# Show stats
uv run python main.py stats --config config/rag_config.yaml
```

## Commands

### Populate Data

Ingests documents from configured directories into the RAG system:

```bash
uv run python main.py populate --config config/rag_config.yaml
```

This command:

* Scans configured directories for documents
* Checks file hashes to skip unchanged files
* Chunks documents according to configuration
* Stores embeddings in ChromaDB with source metadata
* Updates SQLite tracking database

### Clean Stale Documents

Removes documents that no longer exist in the source directories:

```bash
uv run python main.py clean --config config/rag_config.yaml
```

### Interactive Chat

Starts an interactive chat session with the RAG system:

```bash
uv run python main.py chat --config config/rag_config.yaml
```

Features:

* Multi-turn conversations with memory
* Automatic RAG tool selection based on query
* Source citations in responses
* Type 'quit' or 'exit' to end the session

## Configuration

### RAG Domain Configuration

Each RAG domain supports the following options:

```yaml
- name: "domain_name"              # Unique identifier and ChromaDB collection name
  display_name: "Human Name"       # Used in source citations
  directory_path: "path/to/docs"   # Directory containing documents
  description: "Tool description"  # Helps the assistant choose when to use this RAG
  chunk_size: 500                  # Optional: chunk size in characters
  chunk_overlap: 50                # Optional: overlap between chunks
```

### LLM Configuration

The system supports multiple LLM providers through LangChain:

* **OpenAI**: GPT-4, GPT-3.5-turbo, etc.
* **Anthropic**: Claude-3 models
* **Local models**: Via Ollama

Configure via environment variables or modify `src/llm_factory.py`.

## Project Structure

```
├── config/
│   └── rag_config.yaml          # RAG domains configuration
├── data/
│   ├── doc_hashes.db           # SQLite file tracking database
│   └── chroma_db/              # ChromaDB persistence directory
├── docs/                       # Your document directories
│   ├── user_guide/
│   └── infra/
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration loading
│   ├── database.py             # SQLite operations
│   ├── ingestion.py            # Document ingestion pipeline
│   ├── loaders.py              # Document loaders (PDF, Markdown)
│   ├── rag_graph.py           # LangGraph setup and chat functionality
│   └── llm_factory.py         # LLM provider abstraction
├── main.py                     # CLI entry point
├── pyproject.toml             # Project dependencies
├── .env                       # Environment variables
└── README.md
```

## How It Works

### 1\. Ingestion Pipeline

1. **File Discovery**: Scans configured directories for documents
1. **Change Detection**: Uses SQLite to track file hashes and chunking parameters
1. **Chunking**: Processes documents based on type (PDF/Markdown) and configuration
1. **Embedding**: Generates embeddings for new/changed chunks
1. **Storage**: Stores in ChromaDB with rich metadata for source citations

### 2\. Chat System

1. **Query Processing**: User input is processed by the LangGraph assistant
1. **Tool Selection**: Assistant decides which RAG domains to query based on context
1. **Retrieval**: Relevant chunks are retrieved from ChromaDB
1. **Response Generation**: LLM generates response using retrieved context
1. **Source Citations**: Human-readable sources are included in the response

### 3\. Memory Management

* Conversation history is maintained in memory during chat sessions
* Each session is independent
* Memory can be extended to persist across sessions if needed

## Advanced Usage

### Custom LLM Providers

Add new LLM providers by extending `src/llm_factory.py`:

```python
def get_llm(provider: str = None, model: str = None):
    if provider == "your_provider":
        return YourLLMClass(model=model)
    # ... existing providers
```

### Custom Document Loaders

Add support for new document types in `src/loaders.py`:

```python
def load_your_format(file_path: str, chunk_size: int, chunk_overlap: int):
    # Your custom loading logic
    return chunks_with_metadata
```

### Extending RAG Domains

Simply add new entries to your `rag_config.yaml` and run the populate command.

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure your `.env` file contains the required API keys
1. **Empty Responses**: Check that documents were properly ingested with `populate`
1. **Slow Performance**: Consider adjusting chunk sizes or using a different embedding model
1. **Memory Issues**: For large document sets, consider using a more powerful embedding model or reducing chunk sizes

### Debugging

* Check `data/doc_hashes.db` to see which files have been processed
* Use ChromaDB's built-in tools to inspect collections
* Enable debug logging by setting `LOG_LEVEL=DEBUG` in your `.env` file

## Contributing

1. Fork the repository
1. Create a feature branch
1. Make your changes
1. Add tests if applicable
1. Submit a pull request

## License

MIT License - see LICENSE file for details.
