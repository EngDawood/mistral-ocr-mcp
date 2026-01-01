#!/usr/bin/env python3
"""HTTP server for Mistral OCR MCP Server using Smithery HTTP transport."""
import os
import json
import logging
from typing import Optional
from urllib.parse import parse_qs, urlparse

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel, Field

from .server import create_server

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigSchema(BaseModel):
    """Configuration schema for Mistral OCR MCP Server."""
    MISTRAL_API_KEY: str = Field(
        description="Mistral AI API key (get from https://console.mistral.ai/api-keys)"
    )


def parse_config_from_query(query_string: str) -> dict:
    """Parse configuration from query string parameters.
    
    Handles both simple (key=value) and nested (key.nested=value) parameters.
    """
    if not query_string:
        return {}
    
    parsed = parse_qs(query_string)
    config = {}
    
    for key, values in parsed.items():
        value = values[0] if values else ""
        
        # Handle nested keys (e.g., "api.key" -> {"api": {"key": "value"}})
        if "." in key:
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    return config


async def handle_sse(request: Request) -> Response:
    """Handle SSE endpoint for MCP communication."""
    try:
        # Parse configuration from query parameters
        query_string = request.url.query
        config_dict = parse_config_from_query(query_string)
        
        logger.info(f"Received config: {json.dumps(config_dict, indent=2)}")
        
        # Validate configuration
        try:
            config = ConfigSchema(**config_dict)
        except Exception as e:
            logger.error(f"Invalid configuration: {e}")
            return JSONResponse(
                {"error": f"Invalid configuration: {str(e)}"},
                status_code=400
            )
        
        # Set API key in environment for this request
        os.environ["MISTRAL_API_KEY"] = config.MISTRAL_API_KEY
        
        # Create server instance
        mcp_server = create_server()
        
        # Create SSE transport
        sse = SseServerTransport("/mcp/messages")
        
        async def handle_sse_request(request: Request):
            async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send
            ) as streams:
                await mcp_server.run(
                    streams[0],
                    streams[1],
                    mcp_server.create_initialization_options()
                )
        
        return await handle_sse_request(request)
        
    except Exception as e:
        logger.error(f"Error in SSE handler: {e}", exc_info=True)
        return JSONResponse(
            {"error": f"Server error: {str(e)}"},
            status_code=500
        )


async def handle_post(request: Request) -> Response:
    """Handle POST endpoint for MCP messages."""
    try:
        # Parse configuration from query parameters
        query_string = request.url.query
        config_dict = parse_config_from_query(query_string)
        
        logger.info(f"Received config: {json.dumps(config_dict, indent=2)}")
        
        # Validate configuration
        try:
            config = ConfigSchema(**config_dict)
        except Exception as e:
            logger.error(f"Invalid configuration: {e}")
            return JSONResponse(
                {"error": f"Invalid configuration: {str(e)}"},
                status_code=400
            )
        
        # Set API key in environment for this request
        os.environ["MISTRAL_API_KEY"] = config.MISTRAL_API_KEY
        
        # Create server instance
        mcp_server = create_server()
        
        # Create SSE transport
        sse = SseServerTransport("/mcp/messages")
        
        # Handle the POST request
        async def handle_post_request(request: Request):
            async with sse.handle_post_message(
                request.scope,
                request.receive,
                request._send
            ) as streams:
                await mcp_server.run(
                    streams[0],
                    streams[1],
                    mcp_server.create_initialization_options()
                )
        
        return await handle_post_request(request)
        
    except Exception as e:
        logger.error(f"Error in POST handler: {e}", exc_info=True)
        return JSONResponse(
            {"error": f"Server error: {str(e)}"},
            status_code=500
        )


async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "healthy"})


# Create Starlette app
app = Starlette(
    debug=True,
    routes=[
        Route("/health", health_check, methods=["GET"]),
        Route("/mcp", handle_sse, methods=["GET"]),
        Route("/mcp/messages", handle_post, methods=["POST"]),
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id", "mcp-protocol-version"],
)


def main():
    """Run the HTTP server."""
    port = int(os.getenv("PORT", "8081"))
    logger.info(f"Starting MCP HTTP server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()