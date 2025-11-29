# API Reference

## Overview

This API reference documents the Pydantic Stack Demo codebase, which showcases integration patterns for:
- **Pydantic AI**: AI agent framework with type safety and structured outputs
- **Logfire**: Observability and tracing platform
- **FastAPI**: Modern web framework for building APIs
- **Durable Execution**: Integration with DBOS and Temporal for fault-tolerant workflows

The codebase is organized into demonstration modules that illustrate different usage patterns and integration techniques. Each module is self-contained and demonstrates specific capabilities of the Pydantic ecosystem.

## Table of Contents

- [Core Modules](#core-modules)
  - [Logfire Hello World](#logfire-hello-world)
  - [Pydantic AI Hello](#pydantic-ai-hello)
  - [FastAPI Example](#fastapi-example)
- [Agent Patterns](#agent-patterns)
  - [Weather Agent](#weather-agent)
  - [Deep Research Agent](#deep-research-agent)
  - [Twenty Questions Game](#twenty-questions-game)
- [Data Models & Output Types](#data-models--output-types)
  - [Structured Output Extraction](#structured-output-extraction)
  - [Validation with Retry](#validation-with-retry)
- [Memory & Persistence](#memory--persistence)
  - [Message History Memory](#message-history-memory)
  - [Tool-Based Memory](#tool-based-memory)
- [MCP Integration](#mcp-integration)
  - [MCP Client](#mcp-client)
  - [MCP Server](#mcp-server)
- [Durable Execution](#durable-execution)
  - [DBOS Integration](#dbos-integration)
  - [Temporal Integration](#temporal-integration)
- [Distributed Tracing](#distributed-tracing)
  - [Frontend-Backend Example](#frontend-backend-example)
- [Evaluation & Testing](#evaluation--testing)
  - [Model Evaluation](#model-evaluation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Project Dependencies](#project-dependencies)
- [Usage Patterns](#usage-patterns)
- [Best Practices](#best-practices)

---

## Core Modules

### Logfire Hello World

**Source:** `logfire-hello-world/main.py`

Basic Logfire instrumentation demonstrating structured logging capabilities.

#### Functions

##### logfire.configure
```python
def configure(service_name: str = 'hello-world') -> None:
```

Configures Logfire for the application with service identification.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| service_name | str | No | 'hello-world' | Name of the service for telemetry identification |

##### logfire.info
```python
def info(message: str, **kwargs: Any) -> None:
```

Logs an informational message with structured data.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| message | str | Yes | - | Log message with format placeholders |
| **kwargs | Any | No | - | Named arguments to fill message placeholders |

#### Example
```python
import logfire

logfire.configure(service_name='hello-world')
logfire.info('hello {place}', place='world')
```

---

### Pydantic AI Hello

**Source:** `pai-hello/main.py`

Minimal Pydantic AI agent demonstrating basic agent creation and execution.

#### Classes

##### Agent
```python
Agent(
    model: str,
    system_prompt: str | None = None,
    instructions: str | None = None,
)
```

Core agent class for creating AI agents with specific models and behaviors.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | str | Yes | - | Model identifier (e.g., 'openai:gpt-4o') |
| system_prompt | str | No | None | System-level prompt for the agent (deprecated, use instructions) |
| instructions | str | No | None | Instructions defining agent behavior |

##### Methods

###### run_sync
```python
def run_sync(self, prompt: str) -> AgentRunResult:
```

Synchronously executes the agent with a given prompt.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| prompt | str | Yes | - | User prompt to send to the agent |

**Returns:** `AgentRunResult` - Contains output text and execution metadata

#### Example
```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o',
    system_prompt='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
```

---

### FastAPI Example

**Source:** `fastapi-example/main.py`

Demonstrates FastAPI integration with Logfire instrumentation and OpenAI image generation.

#### Application Setup

##### Lifespan Management
```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with AsyncClient() as _http_client:
        http_client = _http_client
        logfire.instrument_httpx(http_client, capture_headers=True)
        yield
```

Manages application lifecycle with proper HTTP client cleanup.

##### FastAPI Application
```python
app = FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app, capture_headers=True)
```

Creates and instruments a FastAPI application with automatic request/response logging.

#### Endpoints

##### GET /
```python
@app.get('/')
@app.get('/display/{image:path}')
async def main() -> FileResponse:
```

Serves the main HTML page for the image generation interface.

**Returns:** `FileResponse` - HTML page file

##### POST /generate
```python
@app.post('/generate')
async def generate_image(prompt: str) -> GenerateResponse:
```

Generates an image using DALL-E 3 and saves it to disk.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| prompt | str | Yes | - | Text prompt for image generation |

**Returns:** `GenerateResponse` - Contains URL to the generated image

#### Data Models

##### GenerateResponse
```python
class GenerateResponse(BaseModel):
    next_url: str = Field(serialization_alias='nextUrl')
```

Response model for image generation endpoint.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| next_url | str | Yes | URL path to view the generated image |

#### Example
```python
import uvicorn
from fastapi import FastAPI
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
logfire.instrument_openai(openai_client)

# Run the application
if __name__ == '__main__':
    uvicorn.run(app)
```

---

## Agent Patterns

### Weather Agent

**Source:** `pai-weather/main.py`

Multi-tool agent demonstrating asynchronous tool calls and dependency injection.

#### Data Models

##### Deps
```python
@dataclass
class Deps:
    client: AsyncClient
```

Dependency container for agent context.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| client | AsyncClient | Yes | HTTP client for making API requests |

##### LatLng
```python
class LatLng(BaseModel):
    lat: float
    lng: float
```

Geographical coordinates model.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| lat | float | Yes | Latitude coordinate |
| lng | float | Yes | Longitude coordinate |

#### Agent Definition

```python
weather_agent = Agent(
    'openai:gpt-4.1-mini',
    instructions='Be concise, reply with one sentence.',
    deps_type=Deps,
    retries=2,
)
```

Agent configured with retry logic and dependency injection.

#### Tools

##### get_lat_lng
**Source:** `pai-weather/main.py:38`

```python
@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
```

Retrieves latitude and longitude for a location description.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| ctx | RunContext[Deps] | Yes | - | Execution context with dependencies |
| location_description | str | Yes | - | Natural language location description |

**Returns:** `LatLng` - Coordinates of the location

##### get_weather
**Source:** `pai-weather/main.py:55`

```python
@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
```

Fetches weather information for given coordinates.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| ctx | RunContext[Deps] | Yes | - | Execution context with dependencies |
| lat | float | Yes | - | Latitude coordinate |
| lng | float | Yes | - | Longitude coordinate |

**Returns:** `dict[str, Any]` - Weather data with temperature and description

#### Example
```python
import asyncio
from httpx import AsyncClient
from pydantic_ai import Agent

async def main():
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)
        result = await weather_agent.run(
            'What is the weather like in London and in Wiltshire?',
            deps=deps
        )
        print('Response:', result.output)

asyncio.run(main())
```

---

### Deep Research Agent

**Source:** `durable-exec/deep_research.py`

Multi-agent workflow demonstrating parallel execution and agent composition.

#### Data Models

##### WebSearchStep
**Source:** `durable-exec/deep_research.py:14`

```python
class WebSearchStep(BaseModel):
    """A step that performs a web search.

    And returns a summary of the search results.
    """
    search_terms: str
```

Represents a single web search operation in a research plan.

##### DeepResearchPlan
**Source:** `durable-exec/deep_research.py:23`

```python
class DeepResearchPlan(BaseModel, **ConfigDict(use_attribute_docstrings=True)):
    """A structured plan for deep research."""

    executive_summary: str
    """A summary of the research plan."""

    web_search_steps: Annotated[list[WebSearchStep], MaxLen(5)]
    """A list of web search steps to perform to gather raw information."""

    analysis_instructions: str
    """The analysis step to perform after all web search steps are completed."""
```

Structured output from the planning agent defining the research strategy.

#### Agents

##### plan_agent
**Source:** `durable-exec/deep_research.py:36`

```python
plan_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Analyze the users query and design a plan for deep research to answer their query.',
    output_type=DeepResearchPlan,
    name='abstract_plan_agent',
)
```

Creates a structured research plan from user query.

##### search_agent
**Source:** `durable-exec/deep_research.py:44`

```python
search_agent = Agent(
    'google-vertex:gemini-2.5-flash',
    instructions='Perform a web search for the given terms and return a detailed report on the results.',
    builtin_tools=[WebSearchTool()],
    name='search_agent',
)
```

Executes web searches using built-in tools.

##### analysis_agent
**Source:** `durable-exec/deep_research.py:51`

```python
analysis_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=AbstractAgent,
    instructions="""
Analyze the research from the previous steps and generate a report on the given subject.

If the search results do not contain enough information, you may perform further searches using the
`extra_search` tool.
""",
    name='analysis_agent',
)
```

Synthesizes search results into a final report with optional follow-up searches.

#### Functions

##### deep_research
**Source:** `durable-exec/deep_research.py:72`

```python
@logfire.instrument
async def deep_research(query: str) -> str:
```

Orchestrates a multi-step research workflow with parallel search execution.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | str | Yes | - | Research question or topic |

**Returns:** `str` - Comprehensive research report

**Workflow:**
1. Generate research plan with `plan_agent`
2. Execute web searches in parallel using `asyncio.TaskGroup`
3. Analyze results with `analysis_agent`
4. Return synthesized report

#### Example
```python
import asyncio

result = await deep_research('Find me a list of hedge funds that write python in London')
print(result)
```

---

### Twenty Questions Game

**Source:** `durable-exec/twenty_questions.py`

Interactive agent-to-agent communication demonstrating tool-based agent interaction.

#### Data Models

##### Answer
**Source:** `durable-exec/twenty_questions.py:12`

```python
class Answer(StrEnum):
    yes = 'yes'
    kind_of = 'kind of'
    not_really = 'not really'
    no = 'no'
    complete_wrong = 'complete wrong'
```

Structured answer types for the question-answer game.

##### GameState
**Source:** `durable-exec/twenty_questions.py:38`

```python
@dataclass
class GameState:
    answer: str
```

Maintains the secret object state across the game.

#### Agents

##### answerer_agent
**Source:** `durable-exec/twenty_questions.py:20`

```python
answerer_agent = Agent(
    'gateway/anthropic:claude-3-5-haiku-latest',
    deps_type=str,
    instructions="""
You are playing a question and answer game.
Your job is to answer questions about a secret object only you know truthfully.
""",
    output_type=Answer,
)
```

Answers questions about a secret object with structured responses.

##### questioner_agent
**Source:** `durable-exec/twenty_questions.py:43`

```python
questioner_agent = Agent(
    'gateway/openai:gpt-4.1',
    deps_type=GameState,
    instructions="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask quantitative questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
)
```

Asks strategic questions to guess the secret object.

#### Tools

##### ask_question
**Source:** `durable-exec/twenty_questions.py:59`

```python
@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Answer:
```

Allows questioner agent to query the answerer agent.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| ctx | RunContext[GameState] | Yes | - | Game state context |
| question | str | Yes | - | Question to ask about the object |

**Returns:** `Answer` - Structured response from answerer agent

#### Functions

##### play
**Source:** `durable-exec/twenty_questions.py:65`

```python
async def play(answer: str) -> AgentRunResult[str]:
```

Executes a complete game session.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| answer | str | Yes | - | The secret object to guess |

**Returns:** `AgentRunResult[str]` - Game results with conversation history

#### Example
```python
import asyncio

result = await play('potato')
print(f'After {len(result.all_messages()) / 2}, the answer is: {result.output}')
```

---

## Data Models & Output Types

### Structured Output Extraction

**Source:** `pai-pydantic/simple.py`

Demonstrates extracting structured data from unstructured text using Pydantic models.

#### Data Models

##### Person
**Source:** `pai-pydantic/simple.py:11`

```python
class Person(BaseModel):
    name: str
    dob: date
    city: str
```

Structured person information extracted from text.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | str | Yes | Person's full name |
| dob | date | Yes | Date of birth |
| city | str | Yes | City of residence |

#### Agent Configuration

```python
agent = Agent(
    'openai:gpt-4.1',
    output_type=Person,
    instructions='Extract information about the person',
)
```

Agent configured to return structured `Person` objects.

#### Example
```python
from datetime import date
from pydantic import BaseModel
from pydantic_ai import Agent

class Person(BaseModel):
    name: str
    dob: date
    city: str

agent = Agent(
    'openai:gpt-4.1',
    output_type=Person,
    instructions='Extract information about the person',
)

result = agent.run_sync("Samuel lived in London and was born on Jan 28th '87")
# result.output is a Person instance
print(result.output.name)  # "Samuel"
print(result.output.city)  # "London"
print(result.output.dob)   # date(1987, 1, 28)
```

---

### Validation with Retry

**Source:** `pai-pydantic/retry.py`

Demonstrates automatic retry on validation failures using Pydantic validators.

#### Data Models

##### Person (with validation)
**Source:** `pai-pydantic/retry.py:11`

```python
class Person(BaseModel):
    name: str
    dob: date
    city: str

    @field_validator('dob')
    def validate_dob(cls, v: date) -> date:
        if v >= date(1900, 1, 1):
            raise ValueError('The person must be born in the 19th century')
        return v
```

Person model with custom validation that triggers automatic retry.

#### Validation Behavior

When validation fails:
1. Agent receives the validation error message
2. Agent automatically retries with corrected data
3. Process continues until validation passes or retries exhausted

#### Example
```python
from datetime import date
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent

class Person(BaseModel):
    name: str
    dob: date
    city: str

    @field_validator('dob')
    def validate_dob(cls, v: date) -> date:
        if v >= date(1900, 1, 1):
            raise ValueError('The person must be born in the 19th century')
        return v

agent = Agent(
    'openai:gpt-4.1',
    output_type=Person,
    instructions='Extract information about the person',
)

# Agent will retry until dob is before 1900
result = agent.run_sync("Samuel lived in London and was born on Jan 28th '87")
# Will interpret as 1887, not 1987
```

---

## Memory & Persistence

### Message History Memory

**Source:** `pai-memory/with_messages.py`

Implements conversational memory using database-stored message history.

#### Database Setup

```python
@asynccontextmanager
async def db() -> AsyncIterator[DbConn]:
    conn = await asyncpg.connect('postgresql://postgres@localhost:5432')
    await conn.execute("""
        create table if not exists messages(
            id serial primary key,
            ts timestamp not null default now(),
            user_id integer not null,
            messages json not null
        )
    """)
    try:
        yield conn
    finally:
        await conn.close()
```

Creates and manages PostgreSQL connection with message storage table.

#### Functions

##### run_agent
**Source:** `pai-memory/with_messages.py:51`

```python
@logfire.instrument
async def run_agent(prompt: str, user_id: int):
```

Executes agent with message history retrieval and storage.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| prompt | str | Yes | - | User input message |
| user_id | int | Yes | - | User identifier for history isolation |

**Workflow:**
1. Retrieve message history from database
2. Run agent with historical context
3. Store new messages to database
4. Return agent response

#### Data Flow

```python
# Retrieve messages
messages: list[ModelMessage] = []
for row in await conn.fetch('SELECT messages FROM messages WHERE user_id = $1 order by ts', user_id):
    messages += ModelMessagesTypeAdapter.validate_json(row[0])

# Run with history
result = await agent.run(prompt, message_history=messages)

# Store new messages
msgs = result.new_messages_json().decode()
await conn.execute('INSERT INTO messages(user_id, messages) VALUES($1, $2)', user_id, msgs)
```

#### Example
```python
import asyncio

async def conversation():
    # First interaction - stores message history
    await run_agent('My name is Samuel.', user_id=123)

    # Second interaction - retrieves history and remembers name
    await run_agent('What is my name?', user_id=123)
    # Response: "Your name is Samuel."

asyncio.run(conversation())
```

---

### Tool-Based Memory

**Source:** `pai-memory/with_tools.py`

Implements agent memory using tools that read/write to a database.

#### Data Models

##### Deps
**Source:** `pai-memory/with_tools.py:46`

```python
@dataclass
class Deps:
    user_id: int
    conn: DbConn
```

Dependencies containing database connection and user context.

#### Database Schema

```sql
create table if not exists memory(
    id serial primary key,
    user_id integer not null,
    value text not null,
    unique(user_id, value)
)
```

#### Agent Configuration

```python
agent = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    instructions='You are a helpful assistant.',
)
```

#### Tools

##### record_memory
**Source:** `pai-memory/with_tools.py:60`

```python
@agent.tool
async def record_memory(ctx: RunContext[Deps], value: str) -> str:
    """Use this tool to store information in memory."""
    await ctx.deps.conn.execute(
        'insert into memory(user_id, value) values($1, $2) on conflict do nothing',
        ctx.deps.user_id,
        value,
    )
    return 'Value added to memory.'
```

Stores information to persistent memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| ctx | RunContext[Deps] | Yes | - | Execution context with user and database |
| value | str | Yes | - | Information to remember |

**Returns:** `str` - Confirmation message

##### retrieve_memories
**Source:** `pai-memory/with_tools.py:71`

```python
@agent.tool
async def retrieve_memories(ctx: RunContext[Deps], memory_contains: str) -> str:
    """Get all memories about the user."""
    rows = await ctx.deps.conn.fetch(
        'select value from memory where user_id = $1 and value ilike $2',
        ctx.deps.user_id,
        f'%{memory_contains}%',
    )
    return '\n'.join(row[0] for row in rows)
```

Retrieves stored memories matching a search query.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| ctx | RunContext[Deps] | Yes | - | Execution context with user and database |
| memory_contains | str | Yes | - | Search string for memory retrieval |

**Returns:** `str` - Newline-separated list of matching memories

#### Example
```python
import asyncio

async def memory_demo():
    async with db(True) as conn:
        deps = Deps(123, conn)

        # Agent automatically uses record_memory tool
        result = await agent.run('My name is Samuel.', deps=deps)
        print(result.output)

    # Later session...
    async with db() as conn:
        deps = Deps(123, conn)

        # Agent automatically uses retrieve_memories tool
        result = await agent.run('What is my name?', deps=deps)
        print(result.output)

asyncio.run(memory_demo())
```

---

## MCP Integration

### MCP Client

**Source:** `pai-mcp-sampling/client.py`

Demonstrates Model Context Protocol (MCP) client integration with sampling.

#### Setup

```python
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(
    command='python',
    args=[str(Path(__file__).parent / 'generate_svg.py')]
)

agent = Agent('openai:gpt-4.1-mini', toolsets=[server])
agent.set_mcp_sampling_model()
```

Creates an MCP server connection and configures agent with MCP sampling.

#### Key Features

- **MCPServerStdio**: Launches MCP server as subprocess
- **set_mcp_sampling_model()**: Enables LLM calls to be proxied through MCP server
- **toolsets**: Registers MCP server tools with the agent

#### Example
```python
import asyncio
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(
    command='python',
    args=[str(Path(__file__).parent / 'generate_svg.py')]
)
agent = Agent('openai:gpt-4.1-mini', toolsets=[server])
agent.set_mcp_sampling_model()

async def main():
    async with agent:
        result = await agent.run('Create an image of a robot in a punk style, it should be pink.')
    print(result.output)

asyncio.run(main())
```

---

### MCP Server

**Source:** `pai-mcp-sampling/generate_svg.py`

MCP server implementation that exposes tools and uses sampling for LLM calls.

#### Server Setup

```python
from mcp.server.fastmcp import FastMCP

app = FastMCP(log_level='WARNING')
```

Creates a FastMCP server application.

#### Agent Definition

```python
svg_agent = Agent(
    instructions='Generate an SVG image as per the user input. Return the SVG data only as a string.'
)
```

Agent used within MCP tools (model provided by client via sampling).

#### Tools

##### image_generator
**Source:** `pai-mcp-sampling/generate_svg.py:20`

```python
@app.tool()
async def image_generator(
    ctx: Context[ServerSessionT, LifespanContextT, RequestT],
    subject: str,
    style: str
) -> str:
```

Generates SVG images using MCP sampling model.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| ctx | Context | Yes | - | MCP context with session information |
| subject | str | Yes | - | Subject of the image |
| style | str | Yes | - | Style of the image |

**Returns:** `str` - Path to generated SVG file

**Implementation:**
```python
# Use MCPSamplingModel to proxy LLM calls through client
svg_result = await svg_agent.run(
    f'{subject=} {style=}',
    model=MCPSamplingModel(ctx.session)
)

# Save SVG to file
path = Path(f'{slugify(subject)}_{slugify(style)}.svg')
path.write_text(svg_result.output)
return f'See {path}'
```

#### Utility Functions

##### slugify
**Source:** `pai-mcp-sampling/generate_svg.py:34`

```python
def slugify(text: str) -> str:
    return re.sub(r'\W+', '-', text.lower())
```

Converts text to filesystem-safe slug.

#### Running the Server

```python
if __name__ == '__main__':
    app.run()  # Runs via stdio
```

---

## Durable Execution

### DBOS Integration

**Source:** `durable-exec/deep_research_dbos.py`

Fault-tolerant workflow execution using DBOS with automatic retry and recovery.

#### Core Components

##### DBOSAgent
```python
from pydantic_ai.durable_exec.dbos import DBOSAgent

dbos_plan_agent = DBOSAgent(plan_agent)
dbos_search_agent = DBOSAgent(search_agent)
dbos_analysis_agent = DBOSAgent(analysis_agent)
```

Wraps Pydantic AI agents for durable execution.

#### Workflows

##### search_workflow
**Source:** `durable-exec/deep_research_dbos.py:83`

```python
@DBOS.workflow()
async def search_workflow(search_terms: str) -> str:
    result = await dbos_search_agent.run(search_terms)
    return result.output
```

Durable workflow for executing a single search.

##### deep_research
**Source:** `durable-exec/deep_research_dbos.py:89`

```python
@DBOS.workflow()
async def deep_research(query: str) -> str:
    result = await dbos_plan_agent.run(query)
    plan = result.output

    # Start parallel workflows
    tasks_handles: List[WorkflowHandleAsync[str]] = []
    for step in plan.web_search_steps:
        task_handle = await DBOS.start_workflow_async(search_workflow, step.search_terms)
        tasks_handles.append(task_handle)

    # Wait for all workflows
    search_results = [await task.get_result() for task in tasks_handles]

    # Analysis
    analysis_result = await dbos_analysis_agent.run(
        format_as_xml({
            'query': query,
            'search_results': search_results,
            'instructions': plan.analysis_instructions,
        }),
    )
    return analysis_result.output
```

Main workflow orchestrating parallel search operations.

#### Configuration

```python
config: DBOSConfig = {
    'name': 'deep_research_durable',
    'enable_otlp': True,
    'system_database_url': 'postgresql://postgres@localhost:5432/dbos',
}
DBOS(config=config)
DBOS.launch()
```

#### Workflow Execution

```python
# Start new workflow
wf_id = f'deep-research-{uuid.uuid4()}'
with SetWorkflowID(wf_id):
    summary = await deep_research(query)

# Resume existing workflow
with SetWorkflowID(existing_wf_id):
    summary = await deep_research(query)
```

#### Example
```python
import asyncio
import sys
from dbos import DBOS, SetWorkflowID

async def run_research():
    config: DBOSConfig = {
        'name': 'deep_research_durable',
        'enable_otlp': True,
        'system_database_url': 'postgresql://postgres@localhost:5432/dbos',
    }
    DBOS(config=config)
    DBOS.launch()

    # Resume workflow if ID provided, otherwise start new
    resume_id = sys.argv[1] if len(sys.argv) > 1 else None
    wf_id = resume_id or f'deep-research-{uuid.uuid4()}'

    with SetWorkflowID(wf_id):
        result = await deep_research(
            'What is the best Python agent framework?'
        )
    print(result)

asyncio.run(run_research())
```

---

### Temporal Integration

**Source:** `durable-exec/deep_research_temporal.py`

Durable execution using Temporal workflows with distributed task execution.

#### Core Components

##### TemporalAgent
```python
from pydantic_ai.durable_exec.temporal import TemporalAgent

temporal_plan_agent = TemporalAgent(plan_agent)
temporal_search_agent = TemporalAgent(search_agent)
temporal_analysis_agent = TemporalAgent(
    analysis_agent,
    activity_config=workflow.ActivityConfig(
        start_to_close_timeout=timedelta(hours=1)
    ),
)
```

Wraps agents for Temporal execution with configurable timeouts.

#### Workflow Definition

##### DeepResearchWorkflow
**Source:** `durable-exec/deep_research_temporal.py:95`

```python
@workflow.defn
class DeepResearchWorkflow:
    @workflow.run
    async def run(self, query: str) -> str:
        result = await temporal_plan_agent.run(query)
        plan = result.output

        # Execute searches in parallel
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(temporal_search_agent.run(step.search_terms))
                for step in plan.web_search_steps
            ]

        search_results = [task.result().output for task in tasks]

        analysis_result = await temporal_analysis_agent.run(
            format_as_xml({
                'query': query,
                'search_results': search_results,
                'instructions': plan.analysis_instructions,
            }),
        )
        return analysis_result.output
```

Temporal workflow orchestrating research process.

#### Worker Configuration

```python
client = await Client.connect(
    'localhost:7233',
    plugins=[PydanticAIPlugin(), LogfirePlugin()]
)

async with Worker(
    client,
    task_queue='deep_research',
    workflows=[DeepResearchWorkflow],
    plugins=[
        AgentPlugin(temporal_plan_agent),
        AgentPlugin(temporal_search_agent),
        AgentPlugin(temporal_analysis_agent),
    ],
):
    # Execute workflow
    summary = await client.execute_workflow(
        DeepResearchWorkflow.run,
        args=[query],
        id=f'deep_research-{uuid.uuid4()}',
        task_queue='deep_research',
    )
```

#### Workflow Resumption

```python
# Resume existing workflow
resume_id = sys.argv[1] if len(sys.argv) > 1 else None
if resume_id:
    summary = await client.get_workflow_handle(resume_id).result()
```

#### Example
```python
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from pydantic_ai.durable_exec.temporal import (
    AgentPlugin,
    LogfirePlugin,
    PydanticAIPlugin
)

async def run_research():
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin(), LogfirePlugin()]
    )

    async with Worker(
        client,
        task_queue='deep_research',
        workflows=[DeepResearchWorkflow],
        plugins=[
            AgentPlugin(temporal_plan_agent),
            AgentPlugin(temporal_search_agent),
            AgentPlugin(temporal_analysis_agent),
        ],
    ):
        result = await client.execute_workflow(
            DeepResearchWorkflow.run,
            args=['What is the best Python agent framework?'],
            id=f'deep_research-{uuid.uuid4()}',
            task_queue='deep_research',
        )
        print(result)

asyncio.run(run_research())
```

---

## Distributed Tracing

### Frontend-Backend Example

**Source:** `distributed-frontend-example/main.py`

Demonstrates distributed tracing across browser and backend using Logfire.

#### Configuration

##### Distributed Tracing
```python
logfire.configure(service_name='backend', distributed_tracing=True)
```

Enables trace propagation across service boundaries.

##### CORS Setup
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],  # Includes 'traceparent' header
)
```

Allows traceparent headers for distributed tracing.

##### Selective Instrumentation
```python
logfire.instrument_fastapi(
    app,
    capture_headers=True,
    excluded_urls='(.+)client-traces$'
)
```

Instruments FastAPI but excludes telemetry proxy endpoint.

#### Endpoints

##### POST /generate
```python
@app.post('/generate')
async def generate_image(prompt: str) -> GenerateResponse:
    response = await openai_client.images.generate(
        prompt=prompt,
        model='dall-e-3'
    )
    # Download and save image
    image_url = response.data[0].url
    r = await http_client.get(image_url)
    path = f'{uuid4().hex}.jpg'
    (image_dir / path).write_bytes(r.content)
    return GenerateResponse(next_url=f'/static/{path}')
```

Image generation with full tracing.

##### POST /client-traces
**Source:** `distributed-frontend-example/main.py:81`

```python
@app.post('/client-traces')
async def client_traces(request: Request):
    """Proxy endpoint for browser telemetry to Logfire."""
    assert logfire_token is not None, 'Logfire token is not set'
    response = await http_client.request(
        method=request.method,
        url=f'{logfire_base_url}v1/traces',
        headers={'Authorization': logfire_token},
        json=await request.json(),
    )
    response.raise_for_status()
    return {
        'status_code': response.status_code,
        'body': response.text,
        'proxied_to': f'{logfire_base_url}v1/traces',
    }
```

Proxies browser traces to Logfire, connecting frontend and backend traces.

#### Environment Variables Required

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| LOGFIRE_TOKEN | str | Yes | Logfire authentication token |
| LOGFIRE_BASE_URL | str | Yes | Logfire API base URL |

#### Example Usage

```python
# Backend setup
import os
os.environ['LOGFIRE_TOKEN'] = 'your-token'
os.environ['LOGFIRE_BASE_URL'] = 'https://logfire-api.pydantic.dev/'

# Run backend
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

# Frontend makes requests with traceparent headers
# Browser telemetry sent to /client-traces endpoint
# All traces connected in Logfire UI
```

---

## Evaluation & Testing

### Model Evaluation

**Source:** `durable-exec/twenty_questions_evals.py`

Systematic evaluation of agent performance across multiple models.

#### Data Models

##### PlayResult
**Source:** `durable-exec/twenty_questions_evals.py:16`

```python
class PlayResult(TypedDict):
    steps: float
    responses: list[Any]
    success: bool
```

Results from a single game evaluation.

| Field | Type | Description |
|-------|------|-------------|
| steps | float | Number of questions asked |
| responses | list[Any] | All agent responses |
| success | bool | Whether game completed successfully |

#### Evaluators

##### QuestionCount
**Source:** `durable-exec/twenty_questions_evals.py:23`

```python
@dataclass
class QuestionCount(Evaluator[str, PlayResult]):
    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> float:
        return ctx.output['steps']
```

Evaluator measuring number of questions asked.

##### QnASuccess
**Source:** `durable-exec/twenty_questions_evals.py:29`

```python
@dataclass
class QnASuccess(Evaluator[str, PlayResult]):
    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> bool:
        return ctx.output['success']
```

Evaluator measuring game completion success rate.

#### Dataset Configuration

```python
dataset: Dataset[str, PlayResult] = Dataset(
    cases=[
        Case(name='Potato', inputs='potato'),
        Case(name='Man', inputs='man'),
        Case(name='Woman', inputs='woman'),
        Case(name='Child', inputs='child'),
        Case(name='Bike', inputs='bike'),
        Case(name='House', inputs='house'),
    ],
    evaluators=[QuestionCount(), QnASuccess()],
)
```

Test cases with different objects to guess.

#### Running Evaluations

##### run_evals
**Source:** `durable-exec/twenty_questions_evals.py:63`

```python
async def run_evals():
    models: list[KnownModelName] = [
        'anthropic:claude-sonnet-4-0',
        'anthropic:claude-sonnet-4-5',
        'openai:gpt-4.1',
        'openai:gpt-4.1-mini',
        'google-vertex:gemini-2.5-flash',
    ]
    for model in models:
        with questioner_agent.override(model=model):
            report = await dataset.evaluate(play_eval, name=f'Q&A {model}')
            report.print(include_input=False, include_output=False)
```

Evaluates multiple models on the same dataset.

#### Example
```python
import asyncio
from pydantic_evals import Dataset, Case

# Define dataset
dataset = Dataset(
    cases=[Case(name='Test1', inputs='potato')],
    evaluators=[QuestionCount(), QnASuccess()],
)

# Run evaluation
async def evaluate():
    with questioner_agent.override(model='openai:gpt-4.1'):
        report = await dataset.evaluate(play_eval, name='GPT-4.1 Eval')
        report.print()

asyncio.run(evaluate())
```

---

## Configuration

### Environment Variables

The following environment variables are used across the codebase:

| Variable | Type | Required | Default | Used In | Description |
|----------|------|----------|---------|---------|-------------|
| OPENAI_API_KEY | str | Yes* | - | All OpenAI integrations | OpenAI API authentication |
| ANTHROPIC_API_KEY | str | Yes* | - | All Anthropic integrations | Anthropic API authentication |
| GOOGLE_API_KEY | str | Yes* | - | Google Vertex integrations | Google Cloud authentication |
| TAVILY_API_KEY | str | Yes* | - | Web search tools | Tavily search API key |
| LOGFIRE_TOKEN | str | No** | - | All examples | Logfire authentication token |
| LOGFIRE_BASE_URL | str | No | - | Distributed tracing | Logfire API endpoint |
| DATABASE_URL | str | Yes*** | - | Memory examples | PostgreSQL connection string |

\* Required when using the corresponding AI provider
\** Optional but recommended for observability
\*** Required only for memory/persistence examples

#### Database Connection

For memory examples, use:
```bash
postgresql://postgres@localhost:5432
```

For DBOS:
```bash
postgresql://postgres@localhost:5432/dbos
```

---

### Project Dependencies

**Source:** `pyproject.toml`

#### Core Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "asyncpg>=0.30.0",           # PostgreSQL async driver
    "devtools>=0.12.2",           # Development utilities
    "fastapi>=0.115.14",          # Web framework
    "logfire[asyncpg,fastapi,httpx]>=4.10",  # Observability platform
    "mcp>=1.15.0",                # Model Context Protocol
    "pydantic-ai>=1.22.0",        # AI agent framework
    "tavily-python>=0.7.12",      # Web search
    "dbos>=2",                    # Durable execution (DBOS)
    "claude-agent-sdk>=0.1.10",   # Claude agent tools
]
```

#### Development Dependencies

```toml
[dependency-groups]
dev = [
    "ruff>=0.12.2",              # Linter and formatter
    "asyncpg-stubs>=0.30.2",     # Type stubs for asyncpg
    "basedpyright>=1.31.6",      # Type checker
]
```

#### Tool Configuration

##### Ruff (Linting & Formatting)

```toml
[tool.ruff]
line-length = 120
target-version = "py39"

[tool.ruff.lint]
extend-select = ["Q", "RUF100", "C90", "UP", "I"]

[tool.ruff.format]
quote-style = "single"
```

##### Pyright (Type Checking)

```toml
[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
venv = ".venv"
exclude = [".git", ".venv", "scratch"]
```

---

## Usage Patterns

### Pattern 1: Basic Agent Creation

Create a simple agent with a single model and instructions:

```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o',
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Your prompt here')
print(result.output)
```

### Pattern 2: Agent with Tools

Add custom tools for external integrations:

```python
from pydantic_ai import Agent, RunContext

@dataclass
class Deps:
    api_key: str

agent = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    instructions='You are a helpful assistant.',
)

@agent.tool
async def search_web(ctx: RunContext[Deps], query: str) -> str:
    """Search the web for information."""
    # Implementation using ctx.deps.api_key
    return search_results

# Run with dependencies
deps = Deps(api_key='your-key')
result = await agent.run('Search for X', deps=deps)
```

### Pattern 3: Structured Output

Extract structured data using Pydantic models:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Person(BaseModel):
    name: str
    age: int
    city: str

agent = Agent(
    'openai:gpt-4o',
    output_type=Person,
    instructions='Extract person information',
)

result = agent.run_sync('John is 30 and lives in NYC')
person: Person = result.output  # Typed as Person
```

### Pattern 4: Multi-Agent Workflow

Compose multiple agents for complex tasks:

```python
planner = Agent('anthropic:claude-sonnet-4-5', output_type=Plan)
executor = Agent('openai:gpt-4o', tools=[tool1, tool2])
analyzer = Agent('anthropic:claude-sonnet-4-5')

# Orchestrate
plan = await planner.run(query)
results = await asyncio.gather(
    *[executor.run(step) for step in plan.output.steps]
)
analysis = await analyzer.run(format_results(results))
```

### Pattern 5: Durable Execution

Make workflows fault-tolerant with DBOS:

```python
from dbos import DBOS
from pydantic_ai.durable_exec.dbos import DBOSAgent

agent = Agent('openai:gpt-4o', tools=[...])
dbos_agent = DBOSAgent(agent)

@DBOS.workflow()
async def process(data: str) -> str:
    result = await dbos_agent.run(data)
    return result.output

# Automatically resumes on failure
with SetWorkflowID('my-workflow'):
    output = await process('input')
```

### Pattern 6: Logfire Instrumentation

Add comprehensive observability:

```python
import logfire

# Configure once at startup
logfire.configure(service_name='my-service')

# Instrument libraries
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(http_client)
logfire.instrument_asyncpg()

# Manual spans
with logfire.span('custom-operation'):
    result = expensive_operation()

# Structured logging
logfire.info('Processing {item_id}', item_id=123)
```

### Pattern 7: Agent Memory

Persistent conversational memory:

```python
# Message history approach
messages = retrieve_messages(user_id)
result = await agent.run(prompt, message_history=messages)
store_messages(user_id, result.new_messages())

# Tool-based approach
@agent.tool
async def remember(ctx: RunContext[Deps], info: str) -> str:
    await ctx.deps.db.store(info)
    return 'Stored'

@agent.tool
async def recall(ctx: RunContext[Deps], query: str) -> str:
    return await ctx.deps.db.search(query)
```

---

## Best Practices

### 1. Type Safety

**Always use type hints and Pydantic models:**
- Define structured outputs with Pydantic models
- Use `deps_type` for dependency injection
- Leverage `output_type` for guaranteed type safety

```python
# Good
class Result(BaseModel):
    score: float
    explanation: str

agent = Agent('openai:gpt-4o', output_type=Result)
result = agent.run_sync(prompt)  # result.output is typed as Result
```

### 2. Error Handling

**Implement retry logic and validation:**
- Use field validators for automatic retry on validation errors
- Set retry counts on agents
- Handle `UsageLimitExceeded` for request limits

```python
class Data(BaseModel):
    value: int

    @field_validator('value')
    def validate_value(cls, v: int) -> int:
        if v < 0:
            raise ValueError('Must be positive')
        return v

agent = Agent('openai:gpt-4o', output_type=Data, retries=3)
```

### 3. Instrumentation

**Enable comprehensive observability:**
- Configure Logfire at application startup
- Instrument all external clients (HTTP, DB, AI)
- Use custom spans for business logic
- Structure log messages with parameters

```python
logfire.configure(service_name='app')
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(http_client)

with logfire.span('business-operation', user_id=user.id):
    result = perform_operation(user)
```

### 4. Resource Management

**Properly manage async resources:**
- Use context managers for database connections
- Implement lifespan handlers for FastAPI
- Close HTTP clients properly
- Use `async with` for agent sessions

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncClient() as client:
        app.state.client = client
        yield

app = FastAPI(lifespan=lifespan)
```

### 5. Agent Design

**Structure agents for clarity and reusability:**
- Single responsibility per agent
- Clear tool descriptions for LLM understanding
- Use `deps` for configuration and state
- Compose agents for complex workflows

```python
# Good: Specialized agents
search_agent = Agent('fast-model', builtin_tools=[WebSearchTool()])
analysis_agent = Agent('smart-model', instructions='Analyze deeply')

# Compose them
search_results = await search_agent.run(query)
analysis = await analysis_agent.run(search_results.output)
```

### 6. Testing & Evaluation

**Systematically evaluate agent performance:**
- Create datasets with diverse test cases
- Define custom evaluators for metrics
- Compare multiple models
- Track performance over time

```python
from pydantic_evals import Dataset, Case

dataset = Dataset(
    cases=[Case(name='Test1', inputs=data1), ...],
    evaluators=[CustomMetric(), SuccessRate()],
)

report = await dataset.evaluate(my_agent_fn)
```

### 7. Durable Execution

**Use for long-running or critical workflows:**
- Wrap agents with `DBOSAgent` or `TemporalAgent`
- Assign workflow IDs for resumption
- Handle interruptions gracefully
- Monitor workflow status

```python
# Critical workflows should be durable
dbos_agent = DBOSAgent(agent)

@DBOS.workflow()
async def critical_task(data: str) -> str:
    # Can resume from any point if interrupted
    result = await dbos_agent.run(data)
    return result.output
```

### 8. Cost & Performance Optimization

**Optimize for cost and latency:**
- Use smaller/faster models for simple tasks
- Implement parallel execution where possible
- Set usage limits to prevent runaway costs
- Cache results when appropriate

```python
# Use appropriate models
fast_agent = Agent('openai:gpt-4.1-mini')  # For simple tasks
smart_agent = Agent('anthropic:claude-sonnet-4-5')  # For complex reasoning

# Parallel execution
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(agent.run(q)) for q in queries]

# Usage limits
result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=10))
```

### 9. Security

**Follow security best practices:**
- Never hardcode API keys
- Use environment variables for secrets
- Validate and sanitize user inputs
- Implement proper CORS for web APIs
- Use authentication for sensitive endpoints

```python
# Good
import os
api_key = os.getenv('OPENAI_API_KEY')

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://trusted-domain.com'],
    allow_credentials=True,
)
```

### 10. Documentation

**Document your agents and tools:**
- Write clear docstrings for tools (visible to LLM)
- Document expected inputs/outputs
- Provide usage examples
- Explain agent roles and capabilities

```python
@agent.tool
async def fetch_data(ctx: RunContext[Deps], query: str) -> dict:
    """Fetch data from the database matching the query.

    Args:
        query: SQL-like query string to filter data

    Returns:
        Dictionary containing matching records and metadata
    """
    return await ctx.deps.db.query(query)
```

---

## See Also

- **Pydantic AI Documentation**: https://ai.pydantic.dev
- **Logfire Documentation**: https://docs.pydantic.dev/logfire
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **DBOS Documentation**: https://docs.dbos.dev
- **Temporal Documentation**: https://docs.temporal.io
- **MCP Protocol**: https://modelcontextprotocol.io
