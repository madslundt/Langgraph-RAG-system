#!/usr/bin/env python3
"""Main CLI entry point for the LangGraph RAG system."""

import os
import sys
from pathlib import Path
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import load_config, ensure_directories
from src.ingestion import DocumentIngestion
from src.graph import RAGChatSystem

# Load environment variables
load_dotenv()

console = Console()


@click.group()
def cli():
    """LangGraph RAG System - A configurable RAG system with LangGraph and LangChain."""
    pass


@cli.command()
@click.option('--config', '-c', default='config/rag_config.yaml',
              help='Path to configuration file')
def populate(config):
    """Populate RAG data from configured directories."""
    try:
        os.makedirs('data', exist_ok=True)

        # Load configuration
        system_config = load_config(config)
        ensure_directories(system_config)

        # Initialize ingestion system
        ingestion = DocumentIngestion(system_config)

        # Populate data
        ingestion.populate_data()

    except Exception as e:
        console.print(f"[red]Error during population: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='config/rag_config.yaml',
              help='Path to configuration file')
def clean(config):
    """Clean stale documents that no longer exist in source directories."""
    try:
        # Load configuration
        system_config = load_config(config)

        # Initialize ingestion system
        ingestion = DocumentIngestion(system_config)

        # Clean stale documents
        ingestion.clean_stale_documents()

    except Exception as e:
        console.print(f"[red]Error during cleaning: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='config/rag_config.yaml',
              help='Path to configuration file')
@click.option('--session-id', '-s', default='default',
              help='Session ID for conversation memory')
def chat(config, session_id):
    """Start interactive chat with the RAG system."""
    try:
        # Load configuration
        system_config = load_config(config)

        # Initialize chat system
        chat_system = RAGChatSystem(system_config)

        # Start interactive chat
        chat_system.start_interactive_chat(session_id)

    except Exception as e:
        console.print(f"[red]Error starting chat: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='config/rag_config.yaml',
              help='Path to configuration file')
def stats(config):
    """Show system statistics."""
    try:
        # Load configuration
        system_config = load_config(config)

        # Initialize ingestion system to get stats
        ingestion = DocumentIngestion(system_config)
        stats_data = ingestion.get_stats()

        # Display database stats
        console.print("[bold blue]Database Statistics[/bold blue]")
        db_table = Table()
        db_table.add_column("Metric", style="cyan")
        db_table.add_column("Value", style="magenta")

        db_table.add_row("Total Files Tracked", str(stats_data['database']['total_files']))

        for domain, count in stats_data['database']['domains'].items():
            db_table.add_row(f"Files in {domain}", str(count))

        console.print(db_table)

    except Exception as e:
        console.print(f"[red]Error getting statistics: {e}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    cli()
