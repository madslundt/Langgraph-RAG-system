"""Document ingestion pipeline for the RAG system."""

import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from .embedding_factory import get_embeddings
from .config import SystemConfig, RAGDomainConfig
from .database import DocumentDatabase
from .loaders import load_document, compute_file_hash


console = Console()


class DocumentIngestion:
    """Handles document ingestion and management."""

    def __init__(self, config: SystemConfig):
        """Initialize ingestion system.

        Args:
            config: System configuration
        """
        self.config = config
        self.embeddings = get_embeddings()
        self.db = DocumentDatabase(config.sqlite_path)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collections for each domain
        self.collections = {}
        for domain in config.rag_domains:
            try:
                # Use LangChain's Chroma wrapper with consistent embedding function
                collection = Chroma(
                    client=self.chroma_client,
                    collection_name=domain.name,
                    embedding_function=self.embeddings,
                )
                self.collections[domain.name] = collection
                console.print(f"[dim]✓ Connected to collection: {domain.name}[/dim]")

            except Exception as e:
                console.print(f"[red]Error initializing collection {domain.name}: {e}[/red]")
                raise

    def populate_data(self) -> None:
        """Populate RAG data from configured directories."""
        console.print("[bold blue]Starting document ingestion...[/bold blue]")

        total_processed = 0
        total_skipped = 0

        for domain in self.config.rag_domains:
            console.print(f"\n[bold green]Processing domain: {domain.display_name}[/bold green]")

            processed, skipped = self._process_domain(domain)
            total_processed += processed
            total_skipped += skipped

        # Display summary
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")

        table.add_row("Files Processed", str(total_processed))
        table.add_row("Files Skipped", str(total_skipped))
        table.add_row("Total Files", str(total_processed + total_skipped))

        console.print(table)
        console.print("[bold green]✅ Document ingestion completed![/bold green]")

    def _process_domain(self, domain: RAGDomainConfig) -> tuple[int, int]:
        """Process all files in a domain directory.

        Returns:
            Tuple of (processed_count, skipped_count)
        """
        if not os.path.exists(domain.directory_path):
            console.print(f"[yellow]Warning: Directory not found: {domain.directory_path}[/yellow]")
            return 0, 0

        # Find all supported files
        files_to_process = []

        if not domain.file_extensions:
            files_to_process.extend(Path(domain.directory_path).glob("**/*"))

        for ext in domain.file_extensions:
            pattern = f"**/*{ext}"
            files_to_process.extend(Path(domain.directory_path).glob(pattern))

        if not files_to_process:
            console.print(f"[yellow]No supported files found in {domain.directory_path}[/yellow]")
            return 0, 0

        processed_count = 0
        skipped_count = 0

        with Progress() as progress:
            task = progress.add_task(f"Processing {domain.name}...", total=len(files_to_process))

            for file_path in files_to_process:
                try:
                    # Get relative path from domain directory
                    relative_path = str(file_path.relative_to(domain.directory_path))

                    if self._should_process_file(file_path, domain, relative_path):
                        self._process_file(file_path, domain, relative_path)
                        processed_count += 1
                        console.print(f"[green]✓[/green] Processed: {relative_path}")
                    else:
                        skipped_count += 1
                        console.print(f"[dim]⏭ Skipped: {relative_path}[/dim]")

                except Exception as e:
                    console.print(f"[red]✗ Error processing {file_path}: {e}[/red]")

                progress.update(task, advance=1)

        return processed_count, skipped_count

    def _should_process_file(self, file_path: Path, domain: RAGDomainConfig,
                           relative_path: str) -> bool:
        """Check if a file should be processed based on hash and chunking parameters.

        Returns:
            True if file should be processed, False if it can be skipped
        """
        # Compute current file hash
        current_hash = compute_file_hash(str(file_path))

        # Get stored file info
        stored_info = self.db.get_file_info(domain.name, relative_path)

        if stored_info is None:
            # File not tracked, needs processing
            return True

        stored_hash, stored_chunk_size, stored_chunk_overlap, _ = stored_info

        # Check if file content or chunking parameters changed
        if (current_hash != stored_hash or
            domain.chunk_size != stored_chunk_size or
            domain.chunk_overlap != stored_chunk_overlap):
            return True

        return False

    def _process_file(self, file_path: Path, domain: RAGDomainConfig,
                     relative_path: str) -> None:
        """Process a single file and update ChromaDB and SQLite.

        Args:
            file_path: Absolute path to the file
            domain: Domain configuration
            relative_path: Path relative to domain directory
        """
        # Load and chunk the document
        domain_dict = {
            'name': domain.name,
            'display_name': domain.display_name,
            'chunk_size': domain.chunk_size,
            'chunk_overlap': domain.chunk_overlap
        }
        try:
            chunks = load_document(str(file_path), domain_dict, relative_path)

            if not chunks:
                console.print(f"[yellow]Warning: No chunks extracted from {relative_path}[/yellow]")
                return
        except ValueError as e:
            print(e)
            return

        collection = self.collections[domain.name]

        # Remove existing chunks for this file
        self._remove_file_chunks(collection, relative_path)

        # Add new chunks
        chunk_ids = []
        documents = []
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "hash": chunk.chunk_hash,
                }
            ))

            chunk_id = f"{relative_path}::{chunk.metadata['chunk_index']}"
            chunk_ids.append(chunk_id)

        # Add chunks to ChromaDB
        collection.add_documents(
            ids=chunk_ids,
            documents=documents,
        )

        # Update SQLite tracking
        file_hash = compute_file_hash(str(file_path))
        self.db.update_file_info(
            domain.name,
            relative_path,
            file_hash,
            domain.chunk_size,
            domain.chunk_overlap
        )

    def _remove_file_chunks(self, collection, file_path: str) -> None:
        """Remove all chunks for a specific file from ChromaDB."""
        try:
            # Query for existing chunks from this file
            results = collection.get(
                where={"file_path": file_path}
            )

            if results['ids']:
                collection.delete(ids=results['ids'])
        except Exception as e:
            # If query fails, it might be because no chunks exist
            console.print(f"[dim]Note: Could not remove existing chunks for {file_path}: {e}[/dim]")

    def clean_stale_documents(self) -> None:
        """Remove documents that no longer exist in the source directories."""
        console.print("[bold blue]Cleaning stale documents...[/bold blue]")

        total_removed = 0

        for domain in self.config.rag_domains:
            console.print(f"\n[bold green]Cleaning domain: {domain.display_name}[/bold green]")

            removed_count = self._clean_domain(domain)
            total_removed += removed_count

        console.print(f"\n[bold green]✅ Cleaned {total_removed} stale documents![/bold green]")

    def _clean_domain(self, domain: RAGDomainConfig) -> int:
        """Clean stale documents for a specific domain.

        Returns:
            Number of files removed
        """
        # Get currently tracked files
        tracked_files = set(self.db.get_tracked_files(domain.name))

        if not tracked_files:
            console.print(f"[dim]No tracked files for domain {domain.name}[/dim]")
            return 0

        # Get currently existing files
        existing_files = set()
        if os.path.exists(domain.directory_path):
            for ext in domain.file_extensions or [""]:
                pattern = f"**/*{ext}"
                for file_path in Path(domain.directory_path).glob(pattern):
                    relative_path = str(file_path.relative_to(domain.directory_path))
                    existing_files.add(relative_path)

        # Find stale files (tracked but no longer existing)
        stale_files = tracked_files - existing_files

        if not stale_files:
            console.print(f"[dim]No stale files found for domain {domain.name}[/dim]")
            return 0

        # Remove stale files
        collection = self.collections[domain.name]

        for file_path in stale_files:
            try:
                # Remove from ChromaDB
                self._remove_file_chunks(collection, file_path)

                # Remove from SQLite
                self.db.remove_file(domain.name, file_path)

                console.print(f"[red]🗑[/red] Removed: {file_path}")

            except Exception as e:
                console.print(f"[red]✗ Error removing {file_path}: {e}[/red]")

        return len(stale_files)

    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        db_stats = self.db.get_stats()

        return {
            'database': db_stats,
        }
