# Architecture Diagrams

## Overview

This codebase demonstrates the Pydantic AI stack through multiple example applications. The architecture is organized into distinct example modules that showcase different capabilities:

- **Pydantic AI Examples**: Core AI agent patterns and integrations
- **Durable Execution Examples**: Agent workflows with DBOS and Temporal
- **API/Frontend Examples**: FastAPI web services with observability
- **Integration Examples**: MCP (Model Context Protocol) and memory patterns

The architecture leverages a layered approach with:
- **Observability Layer**: Logfire for distributed tracing and monitoring
- **Agent Layer**: Pydantic AI agents with tools and structured outputs
- **Integration Layer**: External services (OpenAI, Anthropic, Tavily, PostgreSQL)
- **Execution Layer**: Synchronous, asynchronous, and durable execution patterns

## System Architecture

The system follows a layered architecture with clear separation of concerns. Each example application demonstrates different capabilities while sharing common infrastructure patterns.

```mermaid
graph TB
    subgraph "Observability Layer"
        LF[Logfire]
        LF_OTEL[OTEL/Distributed Tracing]
    end

    subgraph "Application Layer"
        PAI_HELLO[pai-hello<br/>Basic Agent]
        PAI_WEATHER[pai-weather<br/>Tools & Dependencies]
        PAI_PYDANTIC[pai-pydantic<br/>Structured Output]
        PAI_MEMORY[pai-memory<br/>Message History]
        PAI_MCP[pai-mcp-sampling<br/>MCP Integration]
        FASTAPI_EX[fastapi-example<br/>Web Service]
        DIST_FRONT[distributed-frontend<br/>CORS & Proxy]
        DURABLE[durable-exec<br/>Workflows]
    end

    subgraph "Agent Framework Layer"
        PAI[Pydantic AI Core]
        PAI_TOOLS[Agent Tools]
        PAI_MODELS[Model Adapters]
        PAI_DURABLE[Durable Exec<br/>DBOS/Temporal]
        PAI_MCP_CORE[MCP Protocol]
    end

    subgraph "External Services"
        OPENAI[OpenAI<br/>GPT-4]
        ANTHROPIC[Anthropic<br/>Claude]
        GEMINI[Google Vertex<br/>Gemini]
        TAVILY[Tavily<br/>Web Search]
        POSTGRES[PostgreSQL<br/>State/Memory]
    end

    subgraph "Durable Execution Backends"
        DBOS_SVC[DBOS<br/>Workflow Engine]
        TEMPORAL_SVC[Temporal<br/>Workflow Engine]
    end

    %% Application to Framework
    PAI_HELLO --> PAI
    PAI_WEATHER --> PAI
    PAI_WEATHER --> PAI_TOOLS
    PAI_PYDANTIC --> PAI
    PAI_MEMORY --> PAI
    PAI_MEMORY --> POSTGRES
    PAI_MCP --> PAI_MCP_CORE
    FASTAPI_EX --> OPENAI
    DIST_FRONT --> OPENAI
    DURABLE --> PAI_DURABLE

    %% Framework to Services
    PAI_MODELS --> OPENAI
    PAI_MODELS --> ANTHROPIC
    PAI_MODELS --> GEMINI
    PAI_TOOLS --> TAVILY
    PAI_DURABLE --> DBOS_SVC
    PAI_DURABLE --> TEMPORAL_SVC

    %% Observability
    PAI --> LF
    PAI_HELLO --> LF
    PAI_WEATHER --> LF
    PAI_PYDANTIC --> LF
    PAI_MEMORY --> LF
    PAI_MCP --> LF
    FASTAPI_EX --> LF
    DIST_FRONT --> LF_OTEL
    DURABLE --> LF

    style LF fill:#f9f,stroke:#333,stroke-width:2px
    style PAI fill:#bbf,stroke:#333,stroke-width:2px
    style DURABLE fill:#bfb,stroke:#333,stroke-width:2px
```

## Component Relationships

The components interact through well-defined patterns:

1. **Agent-Based Pattern**: Most examples use Pydantic AI Agent as the central component
2. **Dependency Injection**: Agents receive typed dependencies (Deps) for tools
3. **Instrumentation**: Logfire instruments all major components for observability
4. **Async First**: Most implementations use async/await patterns
5. **Type Safety**: Heavy use of Pydantic models for structured data

```mermaid
graph LR
    subgraph "Basic Agent Pattern"
        APP1[Application Code]
        AGENT1[Agent Instance]
        MODEL1[LLM Model]
        APP1 -->|run_sync/run| AGENT1
        AGENT1 -->|prompt| MODEL1
        MODEL1 -->|response| AGENT1
        AGENT1 -->|output| APP1
    end

    subgraph "Agent with Tools Pattern"
        APP2[Application Code]
        AGENT2[Agent Instance]
        DEPS[Dependencies<br/>Dataclass]
        TOOL[Tool Functions]
        MODEL2[LLM Model]
        APP2 -->|run with deps| AGENT2
        AGENT2 -->|context| TOOL
        DEPS -.->|injected| TOOL
        TOOL -->|result| AGENT2
        AGENT2 <-->|function calls| MODEL2
    end

    subgraph "Memory Pattern"
        APP3[Application Code]
        AGENT3[Agent Instance]
        DB[(PostgreSQL)]
        MEM_TOOL[Memory Tools]
        APP3 -->|run with history| AGENT3
        AGENT3 -->|store/retrieve| MEM_TOOL
        MEM_TOOL <-->|SQL| DB
    end

    subgraph "Durable Execution Pattern"
        APP4[Application Code]
        DURABLE_AGENT[DBOSAgent/<br/>TemporalAgent]
        WORKFLOW[Workflow Engine]
        AGENT4[Wrapped Agent]
        APP4 -->|execute| DURABLE_AGENT
        DURABLE_AGENT -->|checkpoint| WORKFLOW
        DURABLE_AGENT -->|delegate| AGENT4
        WORKFLOW -.->|resume on failure| DURABLE_AGENT
    end

    style AGENT1 fill:#bbf,stroke:#333
    style AGENT2 fill:#bbf,stroke:#333
    style AGENT3 fill:#bbf,stroke:#333
    style DURABLE_AGENT fill:#bfb,stroke:#333
```

## Class Hierarchies

The codebase demonstrates extensive use of Pydantic models for data validation and structured outputs.

```mermaid
classDiagram
    class BaseModel {
        <<Pydantic>>
        +model_validate()
        +model_dump()
    }

    class Agent {
        +model: str
        +instructions: str
        +deps_type: Type
        +output_type: Type
        +run(prompt) AgentRunResult
        +run_sync(prompt) AgentRunResult
        +tool() decorator
        +instructions() decorator
    }

    class DBOSAgent {
        +wrapped_agent: Agent
        +run() durable execution
    }

    class TemporalAgent {
        +wrapped_agent: Agent
        +run() temporal workflow
    }

    class MCPServerStdio {
        +command: str
        +args: list
        +provides tools via MCP
    }

    %% Example Data Models
    class Person {
        +name: str
        +dob: date
        +city: str
        +validate_dob()
    }

    class LatLng {
        +lat: float
        +lng: float
    }

    class GenerateResponse {
        +next_url: str
    }

    class Answer {
        <<StrEnum>>
        +yes
        +kind_of
        +not_really
        +no
        +complete_wrong
    }

    class DeepResearchPlan {
        +executive_summary: str
        +web_search_steps: list
        +analysis_instructions: str
    }

    class WebSearchStep {
        +search_terms: str
    }

    %% Dependencies
    class Deps {
        +client: AsyncClient
    }

    class GameState {
        +answer: str
    }

    %% Relationships
    BaseModel <|-- Person
    BaseModel <|-- LatLng
    BaseModel <|-- GenerateResponse
    BaseModel <|-- DeepResearchPlan
    BaseModel <|-- WebSearchStep

    Agent <|-- DBOSAgent : wraps
    Agent <|-- TemporalAgent : wraps

    Agent --> Deps : uses
    Agent --> GameState : uses

    DeepResearchPlan *-- WebSearchStep : contains
```

## Module Dependencies

This diagram shows the import relationships between main modules and their external dependencies.

```mermaid
graph LR

    subgraph "Example Modules"
        direction TB
        LF_HELLO[logfire-hello-world/main.py]
        PAI_HELLO[pai-hello/main.py]
        PAI_WEATHER[pai-weather/main.py]
        PAI_SIMPLE[pai-pydantic/simple.py]
        PAI_RETRY[pai-pydantic/retry.py]
        MEM_MSG[pai-memory/with_messages.py]
        MEM_TOOL[pai-memory/with_tools.py]
        MCP_CLIENT[pai-mcp-sampling/client.py]
        MCP_SVG[pai-mcp-sampling/generate_svg.py]
        FASTAPI_MAIN[fastapi-example/main.py]
        DIST_MAIN[distributed-frontend/main.py]
        TQ[durable-exec/twenty_questions.py]
        TQ_DBOS[durable-exec/twenty_questions_dbos.py]
        TQ_TEMP[durable-exec/twenty_questions_temporal.py]
        TQ_EVAL[durable-exec/twenty_questions_evals.py]
        DR[durable-exec/deep_research.py]
        DR_DBOS[durable-exec/deep_research_dbos.py]
        DR_TEMP[durable-exec/deep_research_temporal.py]
    end

    subgraph "Core Dependencies"
        direction TB
        LOGFIRE[logfire]
        PAI[pydantic_ai]
        PYDANTIC[pydantic]
        ASYNCPG[asyncpg]
        HTTPX[httpx]
        FASTAPI[fastapi]
    end

    subgraph "Tools & Integrations"
        direction TB
        MCP[pydantic_ai.mcp]
        WEBSEARCH[pydantic_ai.WebSearchTool]
        TAVILY_TOOL[pydantic_ai.common_tools.tavily]
        EVALS[pydantic_evals]
    end

    subgraph "Model Providers"
        direction TB
        OPENAI[openai]
    end

    subgraph "Durable Execution"
        direction TB
        DBOS[dbos]
        TEMPORAL[temporalio]
    end

    %% Simple examples
    LF_HELLO --> LOGFIRE

    PAI_HELLO --> PAI
    PAI_HELLO -.-> LOGFIRE

    PAI_WEATHER --> PAI
    PAI_WEATHER --> LOGFIRE
    PAI_WEATHER --> HTTPX
    PAI_WEATHER --> PYDANTIC

    PAI_SIMPLE --> PAI
    PAI_SIMPLE --> LOGFIRE
    PAI_SIMPLE --> PYDANTIC

    PAI_RETRY --> PAI
    PAI_RETRY --> LOGFIRE
    PAI_RETRY --> PYDANTIC

    %% Memory examples
    MEM_MSG --> PAI
    MEM_MSG --> LOGFIRE
    MEM_MSG --> ASYNCPG

    MEM_TOOL --> PAI
    MEM_TOOL --> LOGFIRE
    MEM_TOOL --> ASYNCPG

    %% MCP examples
    MCP_CLIENT --> PAI
    MCP_CLIENT --> LOGFIRE
    MCP_CLIENT --> MCP
    MCP_CLIENT --> MCP_SVG

    %% FastAPI examples
    FASTAPI_MAIN --> FASTAPI
    FASTAPI_MAIN --> LOGFIRE
    FASTAPI_MAIN --> OPENAI
    FASTAPI_MAIN --> HTTPX
    FASTAPI_MAIN --> PYDANTIC

    DIST_MAIN --> FASTAPI
    DIST_MAIN --> LOGFIRE
    DIST_MAIN --> OPENAI
    DIST_MAIN --> HTTPX
    DIST_MAIN --> PYDANTIC

    %% Basic durable exec
    TQ --> PAI
    TQ --> LOGFIRE

    DR --> PAI
    DR --> LOGFIRE
    DR --> WEBSEARCH

    %% DBOS durable exec
    TQ_DBOS --> PAI
    TQ_DBOS --> LOGFIRE
    TQ_DBOS --> DBOS

    DR_DBOS --> PAI
    DR_DBOS --> LOGFIRE
    DR_DBOS --> DBOS
    DR_DBOS --> WEBSEARCH

    %% Temporal durable exec
    TQ_TEMP --> PAI
    TQ_TEMP --> LOGFIRE
    TQ_TEMP --> TEMPORAL

    DR_TEMP --> PAI
    DR_TEMP --> LOGFIRE
    DR_TEMP --> TEMPORAL
    DR_TEMP --> TAVILY_TOOL

    %% Evals
    TQ_EVAL --> PAI
    TQ_EVAL --> LOGFIRE
    TQ_EVAL --> EVALS
    TQ_EVAL --> TQ

    style PAI fill:#bbf,stroke:#333,stroke-width:2px
    style LOGFIRE fill:#f9f,stroke:#333,stroke-width:2px
    style DBOS fill:#bfb,stroke:#333,stroke-width:2px
    style TEMPORAL fill:#bfb,stroke:#333,stroke-width:2px
```

## Data Flow

This diagram illustrates how data flows through different architectural patterns in the codebase.

```mermaid
flowchart LR
    subgraph "Simple Agent Flow (pai-hello)"
        U1[User Code] -->|prompt| A1[Agent]
        A1 -->|API call| M1[LLM Model]
        M1 -->|text response| A1
        A1 -->|AgentRunResult| U1
    end

    subgraph "Tool-Based Flow (pai-weather)"
        U2[User Code] -->|prompt + deps| A2[Agent]
        A2 -->|initial call| M2[LLM Model]
        M2 -->|tool calls| A2
        A2 -->|execute with deps| T1[get_lat_lng Tool]
        A2 -->|execute with deps| T2[get_weather Tool]
        T1 -->|HTTP request| API1[External API]
        T2 -->|HTTP request| API2[External API]
        API1 -->|LatLng| T1
        API2 -->|Weather Data| T2
        T1 -->|result| A2
        T2 -->|result| A2
        A2 -->|tool results| M2
        M2 -->|final answer| A2
        A2 -->|AgentRunResult| U2
    end

    subgraph "Structured Output Flow (pai-pydantic)"
        U3[User Code] -->|prompt| A3[Agent<br/>output_type=Person]
        A3 -->|request| M3[LLM Model]
        M3 -->|JSON response| A3
        A3 -->|validate| V1[Pydantic Validator]
        V1 -.->|validation error| A3
        A3 -.->|retry with error| M3
        V1 -->|valid Person| A3
        A3 -->|Person object| U3
    end

    subgraph "Memory Flow (pai-memory)"
        U4[User Code] -->|prompt + user_id| A4[Agent]
        A4 -->|retrieve| DB[(PostgreSQL)]
        DB -->|message history| A4
        A4 -->|prompt + history| M4[LLM Model]
        M4 -->|response| A4
        A4 -->|store new messages| DB
        A4 -->|AgentRunResult| U4
    end

    subgraph "Durable Execution Flow (DBOS)"
        U5[User Code] -->|start workflow| WF1[DBOS Workflow]
        WF1 -->|execute step| DA1[DBOSAgent]
        DA1 -->|checkpoint state| DBS[(DBOS Database)]
        DA1 -->|run| A5[Agent]
        A5 -->|call| M5[LLM Model]
        M5 -.->|failure| DBS
        DBS -.->|resume| DA1
        M5 -->|success| A5
        A5 -->|result| DA1
        DA1 -->|result| WF1
        WF1 -->|final result| U5
    end

    subgraph "Web API Flow (fastapi-example)"
        CLIENT[Web Client] -->|POST /generate| FE[FastAPI Endpoint]
        FE -->|prompt| OAI[OpenAI API]
        OAI -->|image URL| FE
        FE -->|download| HTTP[HTTPX Client]
        HTTP -->|image bytes| FE
        FE -->|save| FS[File System]
        FE -->|response URL| CLIENT
    end

    style A1 fill:#bbf,stroke:#333
    style A2 fill:#bbf,stroke:#333
    style A3 fill:#bbf,stroke:#333
    style A4 fill:#bbf,stroke:#333
    style DA1 fill:#bfb,stroke:#333
    style WF1 fill:#bfb,stroke:#333
```

## Multi-Agent Orchestration Patterns

The durable-exec examples demonstrate sophisticated multi-agent patterns.

```mermaid
flowchart LR
    subgraph "Twenty Questions Pattern"
        TQ_START[Start Game] --> Q_AGENT[Questioner Agent<br/>Claude Sonnet 4.5]
        Q_AGENT -->|ask_question tool| A_AGENT[Answerer Agent<br/>Claude Haiku]
        A_AGENT -->|Answer enum| Q_AGENT
        Q_AGENT -.->|repeat| Q_AGENT
        Q_AGENT -->|final guess| TQ_END[Game Result]
    end

    subgraph "Deep Research Pattern"
        DR_START[Research Query] --> PLAN_AGENT[Plan Agent<br/>Claude Sonnet 4.5]
        PLAN_AGENT -->|DeepResearchPlan| PARALLEL{Parallel Execution}
        PARALLEL -->|search step 1| SEARCH1[Search Agent 1<br/>Gemini Flash]
        PARALLEL -->|search step 2| SEARCH2[Search Agent 2<br/>Gemini Flash]
        PARALLEL -->|search step N| SEARCHN[Search Agent N<br/>Gemini Flash]
        SEARCH1 -->|results| GATHER[Gather Results]
        SEARCH2 -->|results| GATHER
        SEARCHN -->|results| GATHER
        GATHER -->|all results| ANALYSIS[Analysis Agent<br/>Claude Sonnet 4.5]
        ANALYSIS -.->|extra_search tool| SEARCH_EXTRA[Additional Search<br/>Gemini Flash]
        SEARCH_EXTRA -.->|more data| ANALYSIS
        ANALYSIS -->|final report| DR_END[Research Report]
    end

    subgraph "Durable Execution Wrappers"
        direction LR
        BASIC_AGENT[Agent] -->|wrapped by| DBOS_WRAP[DBOSAgent]
        BASIC_AGENT -->|wrapped by| TEMP_WRAP[TemporalAgent]
        DBOS_WRAP -->|checkpoints to| DBOS_DB[(DBOS DB)]
        TEMP_WRAP -->|checkpoints to| TEMP_DB[(Temporal)]
    end

    style Q_AGENT fill:#bbf,stroke:#333
    style A_AGENT fill:#bbb,stroke:#333
    style PLAN_AGENT fill:#bbf,stroke:#333
    style SEARCH1 fill:#bfb,stroke:#333
    style SEARCH2 fill:#bfb,stroke:#333
    style SEARCHN fill:#bfb,stroke:#333
    style ANALYSIS fill:#bbf,stroke:#333
    style DBOS_WRAP fill:#fbb,stroke:#333
    style TEMP_WRAP fill:#fbb,stroke:#333
```

## Technology Stack Summary

```mermaid
graph LR
    subgraph "Application Examples"
        EX1[8 Example Modules]
    end

    subgraph "Core Framework"
        PAI[Pydantic AI v1.22+]
        PYDANTIC[Pydantic v2]
    end

    subgraph "Observability"
        LOGFIRE[Logfire v4.10+<br/>OTEL Tracing]
    end

    subgraph "Web Framework"
        FASTAPI[FastAPI v0.115+]
        HTTPX[HTTPX AsyncClient]
    end

    subgraph "LLM Providers"
        OPENAI[OpenAI GPT-4/GPT-4.1]
        ANTHROPIC[Anthropic Claude 3.5/4/4.5]
        GEMINI[Google Gemini 2.5 Flash]
    end

    subgraph "Durable Execution"
        DBOS[DBOS v2+]
        TEMPORAL[Temporal Workflows]
    end

    subgraph "Data & Tools"
        POSTGRES[PostgreSQL<br/>asyncpg]
        TAVILY[Tavily Search API]
        MCP[Model Context Protocol]
    end

    subgraph "Evaluation"
        EVALS[Pydantic Evals]
    end

    EX1 --> PAI
    EX1 --> LOGFIRE
    EX1 --> FASTAPI
    PAI --> PYDANTIC
    PAI --> OPENAI
    PAI --> ANTHROPIC
    PAI --> GEMINI
    PAI --> DBOS
    PAI --> TEMPORAL
    PAI --> MCP
    EX1 --> POSTGRES
    EX1 --> TAVILY
    EX1 --> EVALS
    FASTAPI --> HTTPX

    style PAI fill:#bbf,stroke:#333,stroke-width:3px
    style PYDANTIC fill:#bbf,stroke:#333,stroke-width:2px
    style LOGFIRE fill:#f9f,stroke:#333,stroke-width:2px
```

## Legend

### Arrow Types
- `-->` Solid arrow: Direct dependency or data flow
- `-.->` Dashed arrow: Optional/conditional flow or error handling
- `==>` Thick arrow: Primary/critical path

### Node Colors
- Blue: Agent/AI components
- Pink: Observability/monitoring
- Green: Durable execution/workflows
- Grey: Support components
- Red: Wrapper/decorator patterns

### Key Patterns
1. **Agent Pattern**: Core abstraction for LLM interactions
2. **Tool Pattern**: Extend agent capabilities with custom functions
3. **Structured Output**: Type-safe LLM responses via Pydantic
4. **Memory Pattern**: Persistent conversation state
5. **Durable Execution**: Fault-tolerant workflows with checkpointing
6. **Multi-Agent**: Orchestration of multiple specialized agents
7. **Observability**: Comprehensive tracing with Logfire
