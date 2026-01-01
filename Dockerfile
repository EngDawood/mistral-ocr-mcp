# Build stage
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Copy project files
COPY . .

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.16 /uv /bin/uv

# Install dependencies
RUN if [ -f "uv.lock" ]; then \
      echo "Using uv with uv.lock" && \
      export UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy && \
      uv sync --frozen --no-dev; \
    elif [ -f "poetry.lock" ]; then \
      echo "Using poetry with poetry.lock" && \
      export PYTHONUNBUFFERED=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PIP_NO_CACHE_DIR=off \
        PIP_DISABLE_PIP_VERSION_CHECK=on \
        POETRY_HOME="/opt/poetry" \
        POETRY_VIRTUALENVS_IN_PROJECT=true \
        POETRY_NO_INTERACTION=1 && \
      export PATH="$POETRY_HOME/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" && \
      pip install poetry && \
      poetry install --no-dev; \
    else \
      echo "Using uv with pyproject.toml" && \
      export UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy && \
      uv sync --no-dev; \
    fi

# Runtime stage
FROM python:3.12-slim-bookworm AS base

# Copy application from builder
COPY --from=builder /app /app

WORKDIR /app

# Copy uv from builder
COPY --from=ghcr.io/astral-sh/uv:0.5.16 /uv /bin/uv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=8081

# Expose port
EXPOSE 8081

# Start the HTTP server
CMD ["uv", "run", "python", "-m", "mistral_ocr_mcp.http_server"]