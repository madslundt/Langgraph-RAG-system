"""Document loaders for different file types."""

import hashlib
import re
from typing import Any
from pathlib import Path
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentChunk:
    """Represents a chunk of text with metadata."""

    def __init__(self, content: str, metadata: dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.chunk_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the chunk content."""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def load_markdown_file(file_path: str, domain_config: dict[str, Any],
                      relative_path: str) -> list[DocumentChunk]:
    """Load and chunk a Markdown file.

    Args:
        file_path: Absolute path to the file
        domain_config: Domain configuration dictionary
        relative_path: Path relative to domain directory

    Returns:
        List of DocumentChunk objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse markdown to extract sections
    sections = _extract_markdown_sections(content)

    chunks = []
    chunk_size = domain_config.get('chunk_size', 1000)
    chunk_overlap = domain_config.get('chunk_overlap', 100)

    # Create text splitter for large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for section_idx, (heading, section_content) in enumerate(sections):
        # If section is small enough, keep as single chunk
        if len(section_content) <= chunk_size:
            chunk_metadata = {
                'domain': domain_config['name'],
                'file_path': relative_path,
                'chunk_index': section_idx,
                'section_title': heading,
                'section_number': _extract_section_number(heading),
                'source_citation': _generate_markdown_citation(domain_config['display_name'], heading),
                'source_detail': f"{relative_path}:section_{heading}"
            }
            chunks.append(DocumentChunk(section_content, chunk_metadata))
        else:
            # Split large sections into smaller chunks
            section_chunks = text_splitter.split_text(section_content)
            for sub_idx, chunk_content in enumerate(section_chunks):
                chunk_metadata = {
                    'domain': domain_config['name'],
                    'file_path': relative_path,
                    'chunk_index': f"{section_idx}_{sub_idx}",
                    'section_title': heading,
                    'section_number': _extract_section_number(heading),
                    'source_citation': _generate_markdown_citation(domain_config['display_name'], heading),
                    'source_detail': f"{relative_path}:section_{heading}_part_{sub_idx}"
                }
                chunks.append(DocumentChunk(chunk_content, chunk_metadata))

    return chunks


def load_pdf_file(file_path: str, domain_config: dict[str, Any],
                 relative_path: str) -> list[DocumentChunk]:
    """Load and chunk a PDF file.

    Args:
        file_path: Absolute path to the file
        domain_config: Domain configuration dictionary
        relative_path: Path relative to domain directory

    Returns:
        List of DocumentChunk objects
    """
    reader = PdfReader(file_path)
    chunks = []

    chunk_size = domain_config.get('chunk_size', 1500)
    chunk_overlap = domain_config.get('chunk_overlap', 150)

    # Create text splitter for large pages
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()

        if not page_text.strip():
            continue

        # If page is small enough, keep as single chunk
        if len(page_text) <= chunk_size:
            chunk_metadata = {
                'domain': domain_config['name'],
                'file_path': relative_path,
                'chunk_index': page_num - 1,
                'page_number': page_num,
                'source_citation': f"{domain_config['display_name']} page {page_num}",
                'source_detail': f"{relative_path}:page_{page_num}"
            }
            chunks.append(DocumentChunk(page_text, chunk_metadata))
        else:
            # Split large pages into smaller chunks
            page_chunks = text_splitter.split_text(page_text)
            for sub_idx, chunk_content in enumerate(page_chunks):
                chunk_metadata = {
                    'domain': domain_config['name'],
                    'file_path': relative_path,
                    'chunk_index': f"{page_num-1}_{sub_idx}",
                    'page_number': page_num,
                    'source_citation': f"{domain_config['display_name']} page {page_num}",
                    'source_detail': f"{relative_path}:page_{page_num}_part_{sub_idx}"
                }
                chunks.append(DocumentChunk(chunk_content, chunk_metadata))

    return chunks


def _extract_markdown_sections(content: str) -> list[tuple[str, str]]:
    """Extract sections from markdown content based on headers.

    Returns:
        List of (heading, content) tuples
    """
    # Split by headers (# ## ### etc.)
    header_pattern = r'^(#{1,6})\s+(.+)$'
    lines = content.split('\n')

    sections = []
    current_heading = "Introduction"
    current_content = []

    for line in lines:
        header_match = re.match(header_pattern, line, re.MULTILINE)
        if header_match:
            # Save previous section
            if current_content:
                sections.append((current_heading, '\n'.join(current_content).strip()))

            # Start new section
            current_heading = header_match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    # Add final section
    if current_content:
        sections.append((current_heading, '\n'.join(current_content).strip()))

    # Filter out empty sections
    return [(heading, content) for heading, content in sections if content.strip()]


def _extract_section_number(heading: str) -> str:
    """Extract section number from heading if present."""
    # Look for patterns like "1.2.3" at the beginning
    number_pattern = r'^(\d+(?:\.\d+)*)'
    match = re.match(number_pattern, heading.strip())
    return match.group(1) if match else ""


def _generate_markdown_citation(display_name: str, heading: str) -> str:
    """Generate human-readable citation for markdown section."""
    section_number = _extract_section_number(heading)

    if section_number:
        return f"{display_name}: section {section_number}"
    else:
        # Clean up heading for citation
        clean_heading = re.sub(r'^#+\s*', '', heading).strip()
        return f"{display_name}: {clean_heading}"

def load_document(file_path: str, domain_config: dict[str, Any],
                 relative_path: str) -> list[DocumentChunk]:
    """Load a document based on its type.

    Args:
        file_path: Absolute path to the file
        domain_config: Domain configuration dictionary
        relative_path: Path relative to domain directory

    Returns:
        List of DocumentChunk objects

    Raises:
        ValueError: If file type is not supported
    """
    file_extension = Path(file_path).suffix.lower()

    if file_extension in ['.md', '.txt']:
        return load_markdown_file(file_path, domain_config, relative_path)
    elif file_extension == '.pdf':
        return load_pdf_file(file_path, domain_config, relative_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
