# Component Inventory

## Overview

This codebase is a demonstration repository showcasing the Pydantic Stack, including Pydantic AI, Logfire, FastAPI, and integrations with durable execution frameworks (DBOS and Temporal). The project is organized into multiple independent example applications, each demonstrating different capabilities of the Pydantic ecosystem.

The repository contains 10 distinct example applications organized into 7 top-level directories:
- **durable-exec**: Examples of AI agents with durable execution patterns
- **distributed-frontend-example**: Full-stack distributed tracing with FastAPI backend
- **fastapi-example**: Basic FastAPI integration with Logfire and OpenAI
- **logfire-hello-world**: Minimal Logfire instrumentation example
- **pai-hello**: Simplest Pydantic AI agent example
- **pai-mcp-sampling**: MCP (Model Context Protocol) server/client demonstration
- **pai-memory**: Agent memory management patterns (messages and tools)
- **pai-pydantic**: Structured output validation with Pydantic models
- **pai-weather**: Multi-tool agent with dependency injection

## Public API

The project does not expose a traditional library API. Instead, it provides runnable example scripts that demonstrate various patterns and integrations. Each module serves as a standalone entry point for demonstration purposes.

### Modules

#### Distributed Frontend Example
**File**: `distributed-frontend-example/main.py`

A FastAPI backend demonstrating distributed tracing with Logfire, including CORS middleware for frontend integration, image generation via OpenAI DALL-E, and client trace proxying to Logfire.

#### Durable Execution Examples

**File**: `durable-exec/deep_research.py`

Standard async implementation of multi-agent deep research workflow using Pydantic AI agents for planning, searching, and analysis.

**File**: `durable-exec/deep_research_dbos.py`

DBOS-powered durable execution version of deep research workflow with automatic recovery and state persistence.

**File**: `durable-exec/deep_research_temporal.py`

Temporal workflow implementation of deep research with activity-based agent execution and workflow orchestration.

**File**: `durable-exec/twenty_questions.py`

Interactive game demonstrating multi-agent coordination with questioner and answerer agents using usage limits.

**File**: `durable-exec/twenty_questions_dbos.py`

Durable execution version of twenty questions using DBOS workflows for resilient game state management.

**File**: `durable-exec/twenty_questions_evals.py`

Evaluation framework for testing twenty questions agent across multiple AI models with success metrics.

**File**: `durable-exec/twenty_questions_temporal.py`

Temporal workflow-based twenty questions implementation with simulated failures and recovery.

#### FastAPI Integration
**File**: `fastapi-example/main.py`

Basic FastAPI application with Logfire instrumentation, image generation endpoint, and static file serving.

#### Logfire Examples
**File**: `logfire-hello-world/main.py`

Minimal example of Logfire logging with structured data.

#### Pydantic AI Examples

**File**: `pai-hello/main.py`

Simplest possible Pydantic AI agent implementation with synchronous execution.

**File**: `pai-mcp-sampling/client.py`

MCP client demonstrating model context protocol integration with Pydantic AI agents.

**File**: `pai-mcp-sampling/generate_svg.py`

MCP server using FastMCP to provide SVG generation tool via agent sampling.

**File**: `pai-memory/with_messages.py`

Agent memory persistence using PostgreSQL to store and retrieve message history.

**File**: `pai-memory/with_tools.py`

Agent memory management using tool-based approach with record and retrieve capabilities.

**File**: `pai-pydantic/retry.py`

Demonstrates automatic retry on Pydantic validation failures with field validators.

**File**: `pai-pydantic/simple.py`

Basic structured output extraction using Pydantic models as agent output types.

**File**: `pai-weather/main.py`

Multi-tool agent with dependency injection pattern for weather queries.

### Classes

#### Deep Research Models

**DeepResearchPlan** (`deep_research.py:23`, `deep_research_dbos.py:26`, `deep_research_temporal.py:31`)
- Structured plan for deep research workflows
- Attributes: `executive_summary`/`summary`, `web_search_steps`, `analysis_instructions`
- Uses Pydantic's `ConfigDict` with `use_attribute_docstrings=True`

**WebSearchStep** (`deep_research.py:14`, `deep_research_dbos.py:17`, `deep_research_temporal.py:22`)
- Represents a single web search step
- Attribute: `search_terms` (str)

#### Twenty Questions Models

**Answer** (`twenty_questions.py:12`, `twenty_questions_dbos.py:17`, `twenty_questions_temporal.py:52`)
- String enum for question-answer responses
- Values: `yes`, `kind_of`, `not_really`, `no`, `complete_wrong`

**GameState** (`twenty_questions.py:38`, `twenty_questions_dbos.py:46`)
- Dataclass holding game state
- Attribute: `answer` (str) - the secret object

#### Evaluation Models

**PlayResult** (`twenty_questions_evals.py:16`)
- TypedDict for evaluation results
- Fields: `steps` (float), `responses` (list), `success` (bool)

**QuestionCount** (`twenty_questions_evals.py:23`)
- Evaluator for counting question steps
- Returns: float (number of steps)

**QnASuccess** (`twenty_questions_evals.py:29`)
- Evaluator for game success rate
- Returns: bool (success status)

#### API Models

**GenerateResponse** (`distributed-frontend-example/main.py:60`, `fastapi-example/main.py:42`)
- Response model for image generation endpoints
- Field: `next_url` (str) with serialization alias `nextUrl`

**Person** (`pai-pydantic/retry.py:11`, `pai-pydantic/simple.py:11`)
- Structured data extraction model
- Fields: `name` (str), `dob` (date), `city` (str)
- In retry.py: includes validator for 19th century birth dates

**LatLng** (`pai-weather/main.py:32`)
- Geographic coordinates model
- Fields: `lat` (float), `lng` (float)

#### Dependency Models

**Deps** (`pai-memory/with_tools.py:47`, `pai-weather/main.py:18`)
- Dataclass for dependency injection
- In with_tools.py: `user_id` (int), `conn` (DbConn)
- In weather: `client` (AsyncClient)

### Functions

#### Entry Points and Main Functions

**main()** - Multiple implementations across examples
- `distributed-frontend-example/main.py:38-39`: Returns FileResponse for HTML page
- `pai-mcp-sampling/client.py:16-19`: Async function to run MCP agent
- `pai-weather/main.py:79-84`: Async function to execute weather agent

**deep_research()** (`deep_research.py:72`, `deep_research_dbos.py:89`, DBOS workflow)
- Main workflow for deep research pattern
- Parameters: `query` (str)
- Returns: str (analysis result)
- Orchestrates plan, search, and analysis agents

**deep_research_durable()** (`deep_research_dbos.py:112`, `deep_research_temporal.py:117`)
- Durable execution wrapper for deep research
- Parameters: `query` (str)
- Handles workflow initialization and resumption

**play()** (`twenty_questions.py:65`, `twenty_questions_dbos.py:77`, `twenty_questions_temporal.py:73`)
- Main game execution function
- Parameters: `answer` (str), optional `resume_id` (str)
- Returns: AgentRunResult[str] or None
- Manages questioner/answerer agent interaction

**play_eval()** (`twenty_questions_evals.py:47`)
- Evaluation wrapper for play function
- Parameters: `answer` (str)
- Returns: PlayResult
- Handles usage limit exceptions

**run_evals()** (`twenty_questions_evals.py:63`)
- Runs evaluations across multiple AI models
- Tests models: Claude Sonnet 4.0/4.5, GPT-4.1/mini, Gemini 2.5 Flash

**generate_image()** (`distributed-frontend-example/main.py:65`, `fastapi-example/main.py:47`)
- FastAPI endpoint for image generation
- Parameters: `prompt` (str)
- Returns: GenerateResponse
- Generates DALL-E images and serves them locally

**client_traces()** (`distributed-frontend-example/main.py:81`)
- Proxy endpoint for frontend traces to Logfire
- Parameters: `request` (Request)
- Returns: dict with status and proxy info

#### Agent Tool Functions

**ask_question()** (`twenty_questions.py:59`, `twenty_questions_dbos.py:68`, `twenty_questions_temporal.py:52`)
- Tool for questioner agent to query answerer
- Parameters: `ctx` (RunContext), `question` (str)
- Returns: Answer enum or Literal['yes', 'no']

**extra_search()** (`deep_research.py:65`, `deep_research_dbos.py:71`, `deep_research_temporal.py:80`)
- Tool for analysis agent to perform additional searches
- Parameters: `query` (str), optional `ctx` (RunContext)
- Returns: str (search results)

**get_lat_lng()** (`pai-weather/main.py:38`)
- Tool to get geographic coordinates
- Parameters: `ctx` (RunContext[Deps]), `location_description` (str)
- Returns: LatLng

**get_weather()** (`pai-weather/main.py:55`)
- Tool to get weather data for coordinates
- Parameters: `ctx` (RunContext[Deps]), `lat` (float), `lng` (float)
- Returns: dict[str, Any] with temperature and description

**record_memory()** (`pai-memory/with_tools.py:60`)
- Tool to store information in memory database
- Parameters: `ctx` (RunContext[Deps]), `value` (str)
- Returns: str (confirmation message)

**retrieve_memories()** (`pai-memory/with_tools.py:71`)
- Tool to retrieve stored memories
- Parameters: `ctx` (RunContext[Deps]), `memory_contains` (str)
- Returns: str (newline-separated memories)

**image_generator()** (`pai-mcp-sampling/generate_svg.py:20`)
- MCP tool for SVG generation
- Parameters: `ctx` (Context), `subject` (str), `style` (str)
- Returns: str (file path message)

#### Instruction Functions

**add_answer()** (`twenty_questions.py:33`, `twenty_questions_dbos.py:41`)
- Instruction decorator function for answerer agent
- Parameters: `ctx` (RunContext[str])
- Returns: str (formatted secret object)

## Internal Implementation

### Core Modules

#### Workflow Orchestration

**`durable-exec/deep_research.py`**
- Implements multi-agent research workflow pattern
- Uses asyncio.TaskGroup for parallel search execution
- Coordinates plan_agent, search_agent, and analysis_agent
- Demonstrates structured output with XML formatting

**`durable-exec/deep_research_dbos.py`**
- DBOS-based durable execution wrapper
- Uses DBOSAgent wrapper for state persistence
- Implements asynchronous workflow spawning with WorkflowHandleAsync
- Provides workflow resumption by ID

**`durable-exec/deep_research_temporal.py`**
- Temporal workflow implementation with TemporalAgent wrappers
- Uses AgentPlugin for activity registration
- Implements LogfirePlugin and PydanticAIPlugin for observability
- Configures custom activity timeouts

#### Game Implementation

**`durable-exec/twenty_questions.py`**
- Implements question-answer game with two agents
- Uses gateway routing for model selection
- Implements usage limits (25 requests max)
- Demonstrates agent-to-agent communication via tools

**`durable-exec/twenty_questions_dbos.py`**
- Durable version with DBOS state management
- Supports workflow resumption mid-game
- Uses SetWorkflowID for consistent workflow naming

**`durable-exec/twenty_questions_temporal.py`**
- Temporal workflow with simulated failures (10% failure rate)
- Demonstrates automatic retry and recovery
- Uses literal types for yes/no responses

#### Evaluation Framework

**`durable-exec/twenty_questions_evals.py`**
- Imports from twenty_questions module for evaluation
- Defines Dataset with test cases (potato, man, woman, child, bike, house)
- Implements custom evaluators for step count and success rate
- Tests multiple models in parallel

### Helper Classes

#### Workflow Classes

**DeepResearchWorkflow** (`deep_research_temporal.py:95`)
- Temporal workflow definition class
- Method: `run(self, query: str) -> str`
- Orchestrates plan, parallel search, and analysis phases

**TwentyQuestionsWorkflow** (`twenty_questions_temporal.py:66`)
- Temporal workflow definition for game execution
- Method: `run(self) -> None`
- Manages agent interaction lifecycle

### Utility Functions

#### Database Context Managers

**db()** (`pai-memory/with_messages.py:27`, `pai-memory/with_tools.py:27`)
- Async context manager for PostgreSQL connections
- Creates tables if not exists
- In with_messages.py: manages messages table (user_id, messages json)
- In with_tools.py: manages memory table (user_id, value), supports reset parameter

**lifespan()** (`distributed-frontend-example/main.py:30`, `fastapi-example/main.py:20`)
- FastAPI lifespan context manager
- Initializes HTTP client with Logfire instrumentation
- Manages client lifecycle

#### Instrumented Functions

**run_agent()** (`pai-memory/with_messages.py:51`)
- Logfire-instrumented agent execution
- Retrieves message history from database
- Stores new messages after execution
- Parameters: `prompt` (str), `user_id` (int)

**memory_messages()** (`pai-memory/with_messages.py:66`)
- Demonstration sequence for message-based memory
- Shows multi-turn conversation with persistence

**memory_tools()** (`pai-memory/with_tools.py:81`)
- Demonstration sequence for tool-based memory
- Shows information storage and retrieval pattern

#### Utility Functions

**slugify()** (`pai-mcp-sampling/generate_svg.py:34`)
- Converts text to filesystem-safe slug
- Parameters: `text` (str)
- Returns: str (lowercase, non-word chars replaced with hyphens)

## Entry Points

### Command-Line Entry Points

All modules can be executed directly as scripts using the `if __name__ == '__main__':` pattern:

1. **`distributed-frontend-example/main.py:99`**
   - Imports uvicorn and runs FastAPI app
   - Default port from uvicorn configuration

2. **`durable-exec/deep_research.py:94`**
   - Executes: `asyncio.run(deep_research('Find me a list of hedge funds that write python in London'))`

3. **`durable-exec/deep_research_dbos.py:135`**
   - Executes: `asyncio.run(deep_research_durable(...))`
   - Query: 'Whats the best Python agent framework to use if I care about durable execution and type safety?'

4. **`durable-exec/deep_research_temporal.py:145`**
   - Executes: `asyncio.run(deep_research_durable(...))`
   - Same query as DBOS version

5. **`durable-exec/twenty_questions.py:73`**
   - Executes: `asyncio.run(play('potato'))`

6. **`durable-exec/twenty_questions_dbos.py:104`**
   - Executes: `asyncio.run(play(sys.argv[1] if len(sys.argv) > 1 else None, 'potato'))`
   - Supports resumption via command-line argument

7. **`durable-exec/twenty_questions_evals.py:78`**
   - Executes: `asyncio.run(run_evals())`

8. **`durable-exec/twenty_questions_temporal.py:96`**
   - Executes: `asyncio.run(play(sys.argv[1] if len(sys.argv) > 1 else None))`

9. **`fastapi-example/main.py:62`**
   - Imports uvicorn and runs FastAPI app

10. **`logfire-hello-world/main.py:5`**
    - Direct execution: `logfire.info('hello {place}', place='world')`

11. **`pai-hello/main.py:12-13`**
    - Synchronous agent execution: `agent.run_sync('Where does "hello world" come from?')`

12. **`pai-mcp-sampling/client.py:23-25`**
    - Executes: `asyncio.run(main())`

13. **`pai-mcp-sampling/generate_svg.py:39-40`**
    - Runs MCP server: `app.run()`

14. **`pai-memory/with_messages.py:73`**
    - Executes: `asyncio.run(memory_messages())`

15. **`pai-memory/with_tools.py:96`**
    - Executes: `asyncio.run(memory_tools())`

16. **`pai-pydantic/retry.py:28-29`**
    - Synchronous agent execution with validation retry

17. **`pai-pydantic/simple.py:22-23`**
    - Synchronous agent execution with structured output

18. **`pai-weather/main.py:88`**
    - Executes: `asyncio.run(main())`

### FastAPI Endpoints

#### distributed-frontend-example/main.py
- **POST `/generate`** (line 64): Image generation endpoint
- **POST `/client-traces`** (line 80): Logfire trace proxy endpoint
- **GET `/`** (line 36): Serves HTML page
- **GET `/display/{image:path}`** (line 37): Serves HTML page for image display
- **Static mount `/static`** (line 57): Serves generated images

#### fastapi-example/main.py
- **GET `/`** (line 36): Serves HTML page
- **GET `/display/{image:path}`** (line 37): Serves HTML page for image display
- **POST `/generate`** (line 46): Image generation endpoint
- **Static mount `/static`** (line 33): Serves generated images

### MCP Entry Points

#### pai-mcp-sampling/generate_svg.py
- **Tool: `image_generator`** (line 19): MCP tool exposed via FastMCP
- Server runs on stdio transport

## Module Dependency Summary

### Dependency Layers

**Layer 1: Core Infrastructure**
- Logfire: Observability and instrumentation (used by all modules)
- Pydantic: Data validation (used by all modules)
- Pydantic AI: Agent framework (used by all AI examples)

**Layer 2: Integration Frameworks**
- FastAPI: Web framework (distributed-frontend-example, fastapi-example)
- DBOS: Durable execution via PostgreSQL (deep_research_dbos, twenty_questions_dbos)
- Temporal: Workflow orchestration (deep_research_temporal, twenty_questions_temporal)
- MCP: Model Context Protocol (pai-mcp-sampling)

**Layer 3: External Services**
- OpenAI: LLM provider (via pydantic-ai, also direct API for image generation)
- Anthropic: LLM provider (via pydantic-ai)
- Google Vertex: LLM provider (via pydantic-ai)
- PostgreSQL: State persistence (pai-memory, DBOS examples)
- Tavily: Web search (deep_research examples)

### Module Relationships

**Standalone Examples** (no internal dependencies):
- logfire-hello-world
- pai-hello
- pai-pydantic/simple.py
- pai-pydantic/retry.py
- pai-weather
- pai-memory/with_messages.py
- pai-memory/with_tools.py
- fastapi-example
- distributed-frontend-example

**Shared Pattern Implementations**:
- `deep_research.py` → base implementation
  - `deep_research_dbos.py` → DBOS variant
  - `deep_research_temporal.py` → Temporal variant

- `twenty_questions.py` → base implementation
  - `twenty_questions_dbos.py` → DBOS variant
  - `twenty_questions_temporal.py` → Temporal variant
  - `twenty_questions_evals.py` → imports from base for evaluation

**Client-Server Pair**:
- `pai-mcp-sampling/client.py` ↔ `pai-mcp-sampling/generate_svg.py`

### Common Patterns

1. **Agent Definition Pattern**: All AI examples create agents with model specification, instructions, and optional tools
2. **Logfire Instrumentation Pattern**: Configure, then instrument specific integrations (pydantic_ai, httpx, asyncpg, openai, mcp)
3. **Dependency Injection Pattern**: Use dataclasses for deps_type with RunContext for tool access
4. **Durable Execution Pattern**: Wrap agents with DBOSAgent or TemporalAgent for state persistence
5. **Structured Output Pattern**: Define Pydantic models as output_type for validated responses
6. **Message History Pattern**: Store/retrieve via database or pass to run() method
7. **Tool Definition Pattern**: Use @agent.tool or @agent.tool_plain decorators with RunContext

### Configuration

**Project-level** (`pyproject.toml`):
- Python ≥3.12
- Key dependencies: pydantic-ai ≥1.22.0, logfire ≥4.10, fastapi ≥0.115.14
- Development: ruff (formatting/linting), basedpyright (type checking)

**Environment Variables** (implied from code):
- `LOGFIRE_TOKEN`: Required for distributed-frontend-example
- `LOGFIRE_BASE_URL`: Required for distributed-frontend-example
- `TAVILY_API_KEY`: Required for deep_research_temporal
- OpenAI API key: Required for all OpenAI model usage
- Anthropic API key: Required for Claude model usage
- Google Cloud credentials: Required for Vertex AI models

**Database Configuration**:
- PostgreSQL connection string: `postgresql://postgres@localhost:5432` (pai-memory examples)
- DBOS system database: `postgresql://postgres@localhost:5432/dbos` (DBOS examples)
- Temporal server: `localhost:7233` (Temporal examples)
