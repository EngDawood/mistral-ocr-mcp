# Mistral OCR MCP Server

An MCP (Model Context Protocol) server that exposes Mistral AI's OCR and audio transcription capabilities as tools for LLM agents.

## Tools

| Tool | Description |
|------|-------------|
| `pdf_to_text` | Convert PDF to plain text |
| `pdf_to_markdown` | Convert PDF to markdown (preserves formatting) |
| `pdf_from_url` | Download and process PDF from URL |
| `transcribe_audio` | Transcribe audio files to text |

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/mistral-ocr-mcp.git
cd mistral-ocr-mcp

# Install with uv
uv sync

# Run the server
uv run mistral-ocr-mcp
```

### Using pip

```bash
pip install mistral-ocr-mcp
```

## Configuration

Set your Mistral API key:

```bash
# Option 1: Environment variable
export MISTRAL_API_KEY=your_api_key_here

# Option 2: .env file
cp .env.example .env
# Edit .env with your API key
```

Get your free API key from [console.mistral.ai](https://console.mistral.ai/api-keys).

## Usage with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mistral-ocr": {
      "command": "uv",
      "args": ["--directory", "/path/to/mistral-ocr-mcp", "run", "mistral-ocr-mcp"],
      "env": {
        "MISTRAL_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## Usage with Smithery

Install from Smithery:

```bash
npx @smithery/cli install mistral-ocr-mcp
```

## Tool Parameters

### pdf_to_text

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file_path | string | Yes | Path to PDF file |
| pages | string | No | Page selection (e.g., "1,8,9,11-20") |
| output_path | string | No | Custom output path |

### pdf_to_markdown

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file_path | string | Yes | Path to PDF file |
| pages | string | No | Page selection |
| output_path | string | No | Custom output path |

### pdf_from_url

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| url | string | Yes | URL of PDF to download |
| output_format | string | No | "text" or "markdown" (default: "text") |
| pages | string | No | Page selection |
| keep_pdf | boolean | No | Keep downloaded PDF (default: false) |

### transcribe_audio

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file_path | string | Yes | Path to audio file |
| output_path | string | No | Custom output path |

Supported audio formats: `.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac`

## Requirements

- Python 3.10+
- Mistral API key (free tier: 1,000 pages OCR)

## License

MIT
