# Umbrella Audio Transcriber - Architecture Diagrams

## System Architecture

```mermaid
graph TB
    subgraph "External Clients"
        WEB[Web Applications]
        CLI[CLI Tools]
        MCP[MCP/AI Agents]
        N8N[n8n Workflows]
        CIT[Citizen Portal]
    end

    subgraph "API Layer"
        GW[API Gateway<br/>FastAPI]
        AUTH[Auth Middleware]
        RL[Rate Limiter]
        VAL[Request Validator]
    end

    subgraph "Core Processing"
        PQ[Priority Queue]
        TE[Transcription Engine]
        SD[Speaker Diarization]
        SC[Speaker Consolidation]
        EE[Entity Extraction]
        CS[Context System]
    end

    subgraph "Storage"
        REDIS[(Redis<br/>Job Queue)]
        PG[(PostgreSQL<br/>Metadata)]
        S3[(S3/MinIO<br/>Audio & Results)]
    end

    WEB --> GW
    CLI --> GW
    MCP --> GW
    N8N --> GW
    CIT --> GW

    GW --> AUTH
    AUTH --> RL
    RL --> VAL
    VAL --> PQ

    PQ --> TE
    TE --> SD
    SD --> SC
    SC --> EE
    TE --> CS
    
    PQ --> REDIS
    TE --> PG
    EE --> S3
```

## Processing Pipeline

```mermaid
flowchart LR
    subgraph "Input"
        A[Audio File] --> B[Validation]
        B --> C[Duration Check]
    end

    subgraph "Strategy Selection"
        C -->|< 30 min| D[Standard Strategy]
        C -->|> 30 min| E[Chunked Strategy]
    end

    subgraph "Transcription"
        D --> F[Whisper Large-v3]
        E --> F
        F --> G[Raw Transcript]
    end

    subgraph "Diarization"
        G --> H[Pyannote 3.1]
        H --> I[Speaker Segments]
    end

    subgraph "Post-Processing"
        I --> J[Speaker Consolidation]
        J --> K[Entity Extraction]
        K --> L[Quality Metrics]
    end

    subgraph "Output"
        L --> M[Format Result]
        M --> N[Store & Notify]
    end
```

## Priority Queue Flow

```mermaid
graph TB
    subgraph "Job Submission"
        SUB[New Job] --> PRIO{Priority?}
        PRIO -->|Emergency| E[Emergency Queue]
        PRIO -->|Urgent| U[Urgent Queue]
        PRIO -->|Normal| N[Normal Queue]
        PRIO -->|Batch| B[Batch Queue]
        PRIO -->|Citizen| C[Citizen Queue]
    end

    subgraph "Resource Allocation"
        E -->|90% GPU| W1[Worker 1]
        U -->|70% GPU| W2[Worker 2]
        N -->|50% GPU| W3[Worker 3]
        B -->|30% GPU| W4[Worker 4]
        C -->|20% GPU| W5[Worker 5]
    end

    subgraph "Processing"
        W1 --> PROC[Process Job]
        W2 --> PROC
        W3 --> PROC
        W4 --> PROC
        W5 --> PROC
    end
```

## Speaker Consolidation Algorithm

```mermaid
flowchart TD
    A[Speaker Segments] --> B[Build Similarity Matrix]
    B --> C{Context Type?}
    
    C -->|Phone Call| D[Force 2 Speakers]
    C -->|Interview| E[Expect 2 Speakers]
    C -->|Meeting| F[Use Clustering]
    C -->|Legislative| G[Keep All Speakers]
    
    D --> H[Map Speakers]
    E --> H
    F --> H
    G --> H
    
    H --> I[Apply Mapping]
    I --> J[Consolidated Output]
```

## API Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Queue
    participant Worker
    participant Storage
    participant Webhook

    Client->>API: POST /jobs
    API->>API: Validate Request
    API->>Queue: Enqueue Job
    API-->>Client: Return Job ID
    
    Queue->>Worker: Dequeue Job
    Worker->>Worker: Process Audio
    Worker->>Storage: Store Result
    Worker->>Webhook: Notify Completion
    
    Client->>API: GET /jobs/{id}/status
    API-->>Client: Return Status
    
    Client->>API: GET /jobs/{id}/result
    API->>Storage: Fetch Result
    API-->>Client: Return Transcript
```

## Docker Container Architecture

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "umbrella-network"
            TR[Audio Transcriber<br/>:8000]
            RD[Redis<br/>:6379]
            PG[PostgreSQL<br/>:5432]
            NG[Nginx<br/>:80/:443]
        end
        
        subgraph "Volumes"
            V1[model-cache]
            V2[redis-data]
            V3[postgres-data]
            V4[input-files]
            V5[output-files]
        end
    end
    
    subgraph "GPU"
        GPU1[NVIDIA GPU<br/>CUDA 12.1]
    end
    
    TR -.-> V1
    RD -.-> V2
    PG -.-> V3
    TR -.-> V4
    TR -.-> V5
    TR --> GPU1
    
    NG --> TR
    TR --> RD
    TR --> PG
```

## MCP Integration Architecture

```mermaid
graph LR
    subgraph "AI Assistant"
        CLAUDE[Claude/AI Agent]
        MCPC[MCP Client]
    end
    
    subgraph "MCP Server"
        MCPS[MCP Server<br/>Wrapper]
        TOOLS[Tool Registry]
        ASYNC[Async Handler]
    end
    
    subgraph "Transcriber API"
        API[REST API]
        PROC[Processing Engine]
    end
    
    CLAUDE --> MCPC
    MCPC --> MCPS
    MCPS --> TOOLS
    TOOLS --> ASYNC
    ASYNC --> API
    API --> PROC
    
    PROC -.->|Webhook| ASYNC
    ASYNC -.->|Result| MCPC
    MCPC -.->|Response| CLAUDE
```

## Security Layers

```mermaid
graph TB
    subgraph "External"
        CLIENT[Client Request]
    end
    
    subgraph "Edge Security"
        WAF[Web Application Firewall]
        DDOS[DDoS Protection]
        SSL[SSL/TLS Termination]
    end
    
    subgraph "Application Security"
        AUTH[Authentication<br/>API Keys]
        AUTHZ[Authorization<br/>RBAC]
        RATE[Rate Limiting]
        VAL[Input Validation]
    end
    
    subgraph "Data Security"
        ENC[Encryption at Rest]
        CLASS[Classification<br/>Enforcement]
        AUDIT[Audit Logging]
        CHAIN[Chain of Custody]
    end
    
    CLIENT --> WAF
    WAF --> DDOS
    DDOS --> SSL
    SSL --> AUTH
    AUTH --> AUTHZ
    AUTHZ --> RATE
    RATE --> VAL
    VAL --> ENC
    ENC --> CLASS
    CLASS --> AUDIT
    AUDIT --> CHAIN
```

## Performance Optimization Flow

```mermaid
flowchart LR
    subgraph "Input Optimization"
        A1[Audio Input] --> A2[Format Detection]
        A2 --> A3[Resampling<br/>16kHz]
        A3 --> A4[Mono Conversion]
    end
    
    subgraph "GPU Optimization"
        B1[Model Loading] --> B2[Quantization<br/>INT8/FP16]
        B2 --> B3[Batch Processing]
        B3 --> B4[Memory Pool]
    end
    
    subgraph "Caching"
        C1[L1: Memory<br/>LRU Cache]
        C2[L2: Redis<br/>Distributed]
        C3[L3: Disk<br/>Persistent]
        C1 --> C2
        C2 --> C3
    end
    
    subgraph "Output"
        D1[Result Assembly] --> D2[Compression]
        D2 --> D3[Storage]
    end
    
    A4 --> B1
    B4 --> C1
    C3 --> D1
```

## Legislative Processing Specialization

```mermaid
graph TD
    subgraph "Audio Input"
        AUDIO[Legislative Audio]
    end
    
    subgraph "Pre-Processing"
        GAVEL[Gavel Detection]
        ROLL[Roll Call Markers]
        FORMAL[Formal Speech Patterns]
    end
    
    subgraph "Entity Extraction"
        BILLS[Bill Numbers<br/>SB/HB Patterns]
        LEGS[Legislator Names]
        COMMS[Committees]
        VOTES[Vote Tallies]
    end
    
    subgraph "Post-Processing"
        STRUCT[Structure Detection<br/>Agenda Items]
        TIME[Timeline Creation]
        SUMM[Motion Summaries]
    end
    
    AUDIO --> GAVEL
    AUDIO --> ROLL
    AUDIO --> FORMAL
    
    GAVEL --> BILLS
    ROLL --> LEGS
    FORMAL --> COMMS
    
    BILLS --> STRUCT
    LEGS --> TIME
    COMMS --> SUMM
    VOTES --> SUMM
```

## Error Recovery Flow

```mermaid
stateDiagram-v2
    [*] --> Processing
    Processing --> Success: Complete
    Processing --> Error: Exception
    
    Error --> Analyze: Error Type
    
    Analyze --> RetryableError: Transient
    Analyze --> FatalError: Permanent
    
    RetryableError --> Retry: Attempt < 3
    RetryableError --> Failed: Attempt >= 3
    
    Retry --> Processing: Backoff
    
    FatalError --> Failed: Log & Alert
    Failed --> Cleanup: Release Resources
    Success --> Cleanup: Release Resources
    
    Cleanup --> [*]
```

## Monitoring Dashboard Layout

```mermaid
graph TB
    subgraph "Metrics Dashboard"
        subgraph "Real-Time"
            RT1[Active Jobs]
            RT2[Queue Depth]
            RT3[GPU Usage]
            RT4[Processing Speed]
        end
        
        subgraph "Performance"
            P1[Avg Processing Time]
            P2[Success Rate]
            P3[Error Rate]
            P4[Throughput]
        end
        
        subgraph "Resources"
            R1[CPU Usage]
            R2[Memory Usage]
            R3[Disk I/O]
            R4[Network I/O]
        end
        
        subgraph "Business"
            B1[Jobs by Priority]
            B2[Jobs by Type]
            B3[Cost Analysis]
            B4[User Statistics]
        end
    end
```

These diagrams provide a comprehensive visual representation of the Umbrella Audio Transcriber's architecture, workflows, and key components. They can be rendered using any Mermaid-compatible viewer or documentation system.