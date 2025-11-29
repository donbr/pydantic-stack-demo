# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository demonstrates the Pydantic ecosystem stack: **PydanticAI** for building AI agents, **Logfire** for observability, and integration with durable execution frameworks (Temporal, DBOS).

## Commands

```bash
# Install dependencies (uses uv)
make install

# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Run individual examples
uv run python pai-hello/main.py
uv run python pai-weather/main.py
uv run python fastapi-example/main.py
uv run python durable-exec/deep_research.py

# Run with durable execution (requires Temporal or DBOS)
uv run python durable-exec/deep_research_temporal.py
uv run python durable-exec/deep_research_dbos.py

# Resume a workflow by ID
uv run python durable-exec/deep_research_temporal.py <workflow-id>
```

## Code Style

- Uses **ruff** for formatting and linting (single quotes, 120 char line length)
- Uses **basedpyright** for strict type checking
- Target Python 3.12+, strict type checking mode
- Import style: combine-as-imports enabled, isort-compatible

## Architecture

### Example Directories

- `pai-hello/` - Minimal PydanticAI agent example
- `pai-weather/` - Agent with tools, dependencies, and httpx integration
- `pai-pydantic/` - Structured output with Pydantic models
- `pai-memory/` - Agent memory with PostgreSQL persistence
- `pai-mcp-sampling/` - MCP server/client with sampling
- `fastapi-example/` - FastAPI + Logfire + OpenAI image generation
- `logfire-hello-world/` - Basic Logfire setup
- `distributed-frontend-example/` - React frontend with distributed tracing

### Durable Execution (`durable-exec/`)

Three implementations of the same multi-agent deep research pattern:
- `deep_research.py` - Base implementation without durability
- `deep_research_temporal.py` - Uses Temporal for durable execution
- `deep_research_dbos.py` - Uses DBOS for durable execution

Pattern: Plan agent → parallel search agents → analysis agent with extra search tool.

### Generated Documentation (`architecture/`)

Auto-generated comprehensive documentation suite for the repository:

- `architecture/README.md` - Main overview with technology stack, patterns, and navigation guide
- `architecture/docs/01_component_inventory.md` - Complete catalog of modules, classes, and functions
- `architecture/diagrams/02_architecture_diagrams.md` - Mermaid diagrams showing system structure
- `architecture/docs/03_data_flows.md` - Sequence diagrams for execution patterns
- `architecture/docs/04_api_reference.md` - Detailed API documentation with examples

Key patterns documented: Simple Agent, Tool-Based Agent, Structured Output, Memory Pattern, Multi-Agent, Durable Execution, MCP Integration.

## Key Dependencies

- `pydantic-ai` - AI agent framework with structured outputs
- `logfire` - Observability with AsyncPG, FastAPI, HTTPX instrumentation
- `temporalio` / `dbos` - Durable execution backends
- `mcp` - Model Context Protocol
- `tavily-python` - Web search tool

## Environment Variables

Required API keys vary by example:
- `OPENAI_API_KEY` - For OpenAI models
- `ANTHROPIC_API_KEY` - For Claude models
- `TAVILY_API_KEY` - For web search tools
- `LOGFIRE_TOKEN` - For Logfire observability

## Running Services

```bash
# PostgreSQL for memory examples
docker run -e POSTGRES_HOST_AUTH_METHOD=trust --rm -it --name pg -p 5432:5432 -d postgres

# Temporal for durable execution
temporal server start-dev

# FastAPI example
uv run python fastapi-example/main.py
```
