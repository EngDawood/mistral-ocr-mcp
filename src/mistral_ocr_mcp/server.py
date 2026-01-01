#!/usr/bin/env python3
"""
Mistral OCR MCP Server

An MCP server exposing Mistral AI's OCR and audio transcription capabilities.

Tools:
- pdf_to_text: Convert PDF files to plain text
- pdf_to_markdown: Convert PDF files to markdown
- pdf_from_url: Download and process PDF from URL
- transcribe_audio: Transcribe audio files to text
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mistralai import DocumentURLChunk, Mistral
from mistralai.models.file import File

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("mistral_ocr_mcp")

# Constants
DEFAULT_OCR_MODEL = "mistral-ocr-latest"
DEFAULT_AUDIO_MODEL = "voxtral-mini-latest"


# --- Helper Functions ---

def _log(message: str) -> None:
    """Log to stderr for debugging."""
    print(message, file=sys.stderr)


def _get_api_key() -> str:
    """Get Mistral API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY not found. Set it in your environment or .env file."
        )
    return api_key


def _parse_page_spec(page_spec: str) -> set[int]:
    """Parse page specification string into a set of page numbers.

    Args:
        page_spec: String like "1,8,9,11-20" or "1-5,10"

    Returns:
        Set of page numbers (1-indexed)
    """
    pages = set()
    parts = page_spec.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            range_parts = part.split("-")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid page range format: '{part}'")
            start = int(range_parts[0].strip())
            end = int(range_parts[1].strip())
            if start < 1 or end < 1:
                raise ValueError("Page numbers must be positive")
            if start > end:
                raise ValueError(f"Invalid page range: {start}-{end}")
            pages.update(range(start, end + 1))
        else:
            page_num = int(part)
            if page_num < 1:
                raise ValueError("Page numbers must be positive")
            pages.add(page_num)

    return pages


def _markdown_to_text(content: str) -> str:
    """Strip markdown formatting to produce plain text."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", content)  # Remove images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # Keep link text
    text = re.sub(r"[#*_`~]+", "", text)  # Remove emphasis markers
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _download_pdf_from_url(url: str, output_dir: Path) -> Path:
    """Download a PDF file from a URL."""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    if not filename or not filename.lower().endswith(".pdf"):
        filename = "downloaded_document.pdf"

    output_path = output_dir / filename

    with urlopen(url) as response:
        pdf_data = response.read()

    output_path.write_bytes(pdf_data)
    return output_path


def _process_pdf_ocr(
    pdf_path: Path,
    to_text: bool = True,
    page_numbers: Optional[set[int]] = None,
    output_path: Optional[Path] = None
) -> tuple[Path, int]:
    """Process PDF with Mistral OCR.

    Args:
        pdf_path: Path to the PDF file
        to_text: If True, convert to plain text; if False, keep markdown
        page_numbers: Set of page numbers to process (1-indexed)
        output_path: Custom output path

    Returns:
        Tuple of (output_path, page_count)
    """
    pdf_path = pdf_path.expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        ext = ".txt" if to_text else ".md"
        output_path = pdf_path.with_suffix(ext)

    api_key = _get_api_key()
    client = Mistral(api_key=api_key)

    # Upload file
    file_bytes = pdf_path.read_bytes()
    uploaded = client.files.upload(
        file={"file_name": pdf_path.name, "content": file_bytes},
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded.id, expiry=1)

    # Process OCR
    response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model=DEFAULT_OCR_MODEL,
        include_image_base64=False,
    )

    # Filter pages if specified
    if page_numbers:
        filtered_pages = []
        for idx, page in enumerate(response.pages, start=1):
            if idx in page_numbers:
                filtered_pages.append(page.markdown)
        markdown_pages = filtered_pages
        page_count = len(filtered_pages)
    else:
        markdown_pages = [page.markdown for page in response.pages]
        page_count = len(response.pages)

    markdown_content = "\n\n".join(markdown_pages)

    # Convert to plain text if requested
    if to_text:
        final_content = _markdown_to_text(markdown_content)
    else:
        final_content = markdown_content

    output_path.write_text(final_content, encoding="utf-8")
    return output_path, page_count


def _transcribe_audio_file(audio_path: Path, output_path: Optional[Path] = None) -> Path:
    """Transcribe audio file using Mistral Voxtral.

    Args:
        audio_path: Path to the audio file
        output_path: Custom output path

    Returns:
        Path to the transcription file
    """
    audio_path = Path(audio_path).expanduser().resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if output_path is None:
        output_path = audio_path.with_suffix(".txt")

    api_key = _get_api_key()

    with Mistral(api_key=api_key) as client:
        with open(audio_path, "rb") as f:
            file_data = f.read()
            file_obj = File(file_name=audio_path.name, content=file_data)
            response = client.audio.transcriptions.complete(
                model=DEFAULT_AUDIO_MODEL,
                file=file_obj
            )

    output_path.write_text(response.text, encoding="utf-8")
    return output_path


def _format_success(data: dict) -> str:
    """Format successful response as JSON."""
    return json.dumps({"success": True, **data}, indent=2)


def _format_error(error: str) -> str:
    """Format error response as JSON."""
    return json.dumps({"success": False, "error": error}, indent=2)


# --- MCP Tools ---

@mcp.tool(
    name="pdf_to_text",
    annotations={
        "title": "Convert PDF to Plain Text",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def pdf_to_text(
    file_path: str,
    pages: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """Convert a PDF file to plain text using Mistral OCR.

    Processes the PDF and extracts text content, stripping markdown formatting.
    Supports processing specific pages.

    Args:
        file_path: Path to the PDF file to convert
        pages: Specific pages to process (e.g., '1,8,9,11-20'). If not specified, all pages are processed.
        output_path: Custom output path. Defaults to same directory as input with .txt extension.

    Returns:
        JSON with output_path and page_count
    """
    try:
        pdf_path = Path(file_path)
        if not file_path.lower().endswith(".pdf"):
            return _format_error("File must be a PDF (.pdf extension)")

        page_numbers = _parse_page_spec(pages) if pages else None
        out_path = Path(output_path) if output_path else None

        result_path, page_count = _process_pdf_ocr(
            pdf_path=pdf_path,
            to_text=True,
            page_numbers=page_numbers,
            output_path=out_path
        )

        return _format_success({
            "output_path": str(result_path),
            "page_count": page_count,
            "format": "text"
        })

    except Exception as e:
        return _format_error(str(e))


@mcp.tool(
    name="pdf_to_markdown",
    annotations={
        "title": "Convert PDF to Markdown",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def pdf_to_markdown(
    file_path: str,
    pages: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """Convert a PDF file to markdown using Mistral OCR.

    Processes the PDF and preserves markdown formatting including headings,
    tables, and figure references. Supports processing specific pages.

    Args:
        file_path: Path to the PDF file to convert
        pages: Specific pages to process (e.g., '1,8,9,11-20'). If not specified, all pages are processed.
        output_path: Custom output path. Defaults to same directory as input with .md extension.

    Returns:
        JSON with output_path and page_count
    """
    try:
        pdf_path = Path(file_path)
        if not file_path.lower().endswith(".pdf"):
            return _format_error("File must be a PDF (.pdf extension)")

        page_numbers = _parse_page_spec(pages) if pages else None
        out_path = Path(output_path) if output_path else None

        result_path, page_count = _process_pdf_ocr(
            pdf_path=pdf_path,
            to_text=False,
            page_numbers=page_numbers,
            output_path=out_path
        )

        return _format_success({
            "output_path": str(result_path),
            "page_count": page_count,
            "format": "markdown"
        })

    except Exception as e:
        return _format_error(str(e))


@mcp.tool(
    name="pdf_from_url",
    annotations={
        "title": "Process PDF from URL",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def pdf_from_url(
    url: str,
    output_format: str = "text",
    pages: Optional[str] = None,
    keep_pdf: bool = False
) -> str:
    """Download and process a PDF from a URL using Mistral OCR.

    Downloads the PDF, processes it, and optionally deletes the downloaded file.

    Args:
        url: URL of the PDF file to download and process
        output_format: Output format: 'text' for plain text or 'markdown' for markdown
        pages: Specific pages to process (e.g., '1,8,9,11-20'). If not specified, all pages are processed.
        keep_pdf: Keep the downloaded PDF file after processing

    Returns:
        JSON with output_path, page_count, and pdf_path (if kept)
    """
    try:
        if output_format.lower() not in ("text", "markdown"):
            return _format_error("output_format must be 'text' or 'markdown'")

        # Download PDF to temp directory
        download_dir = Path(tempfile.gettempdir())
        pdf_path = _download_pdf_from_url(url, download_dir)

        page_numbers = _parse_page_spec(pages) if pages else None
        to_text = output_format.lower() == "text"

        result_path, page_count = _process_pdf_ocr(
            pdf_path=pdf_path,
            to_text=to_text,
            page_numbers=page_numbers
        )

        response_data = {
            "output_path": str(result_path),
            "page_count": page_count,
            "format": output_format.lower()
        }

        # Clean up or report PDF path
        if keep_pdf:
            response_data["pdf_path"] = str(pdf_path)
        else:
            pdf_path.unlink()
            response_data["pdf_deleted"] = True

        return _format_success(response_data)

    except Exception as e:
        return _format_error(str(e))


@mcp.tool(
    name="transcribe_audio",
    annotations={
        "title": "Transcribe Audio to Text",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def transcribe_audio(
    file_path: str,
    output_path: Optional[str] = None
) -> str:
    """Transcribe an audio file to text using Mistral Voxtral.

    Supports multiple audio formats including .ogg, .mp3, .wav, .m4a, .flac.
    Handles multiple languages including Arabic and English.

    Args:
        file_path: Path to the audio file to transcribe (supports .ogg, .mp3, .wav, .m4a, .flac)
        output_path: Custom output path. Defaults to same directory as input with .txt extension.

    Returns:
        JSON with output_path
    """
    try:
        audio_path = Path(file_path)
        out_path = Path(output_path) if output_path else None

        result_path = _transcribe_audio_file(
            audio_path=audio_path,
            output_path=out_path
        )

        return _format_success({
            "output_path": str(result_path)
        })

    except Exception as e:
        return _format_error(str(e))


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
