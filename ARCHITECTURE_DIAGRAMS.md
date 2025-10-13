# Data Science Koans - Architecture Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "Learning Experience"
        USER[👤 Learner]
        JN[📓 Jupyter Notebook]
        PROGRESS[📊 Progress Dashboard]
    end
    
    subgraph "Core Framework"
        VALIDATOR[✅ KoanValidator]
        TRACKER[📈 ProgressTracker]
        DATAGEN[🎲 DataGenerator]
    end
    
    subgraph "Storage"
        JSON[(progress.json)]
        SOLUTIONS[💡 Solutions]
    end
    
    USER --> JN
    JN --> VALIDATOR
    JN --> DATAGEN
    VALIDATOR --> TRACKER
    TRACKER --> JSON
    TRACKER --> PROGRESS
    SOLUTIONS -.reference.-> USER
```

## Learning Path Flow

```mermaid
graph LR
    START[Start] --> L1[Level 1: Foundation]
    L1 --> L2[Level 2: Data Prep]
    L2 --> L3[Level 3: Modeling]
    L3 --> L4[Level 4: Advanced]
    L4 --> L5[Level 5: Best Practices]
    L5 --> COMPLETE[🎓 Complete!]
    
    style START fill:#e1f5e1
    style COMPLETE fill:#ffe1e1
    style L1 fill:#e3f2fd
    style L2 fill:#fff3e0
    style L3 fill:#f3e5f5
    style L4 fill:#fce4ec
    style L5 fill:#e0f2f1
```

## Koan Execution Flow

```mermaid
sequenceDiagram
    participant L as Learner
    participant N as Notebook
    participant V as Validator
    participant P as ProgressTracker
    
    L->>N: Read koan explanation
    L->>N: Write solution in TODO
    L->>N: Run validation cell
    N->>V: Call validation function
    
    alt Solution Correct
        V->>V: Run assertions (pass)
        V->>P: Mark koan complete
        V->>N: Display ✅ SUCCESS
        P->>P: Update mastery levels
        N->>L: Show next koan
    else Solution Incorrect
        V->>V: Assertion fails
        V->>N: Display ❌ FAILED + hint
        N->>L: Review and retry
    end
```

## Module Organization

```mermaid
graph TD
    ROOT[datascience-koans]
    
    ROOT --> KOANS[koans/]
    ROOT --> DATA[data/]
    ROOT --> TESTS[tests/]
    ROOT --> DOCS[memory-bank/]
    
    KOANS --> CORE[core/]
    KOANS --> NOTEBOOKS[notebooks/]
    KOANS --> SOLUTIONS[solutions/]
    
    CORE --> VAL[validator.py]
    CORE --> PROG[progress.py]
    CORE --> GEN[data_gen.py]
    
    NOTEBOOKS --> NB01[01_numpy_fundamentals.ipynb]
    NOTEBOOKS --> NB02[02_pandas_essentials.ipynb]
    NOTEBOOKS --> NB03[...]
    NOTEBOOKS --> NB15[15_ethics_and_bias.ipynb]
    
    DATA --> PJSON[progress.json]
    
    style ROOT fill:#e1f5e1
    style KOANS fill:#e3f2fd
    style CORE fill:#fff3e0
    style NOTEBOOKS fill:#f3e5f5
```

## Content Progression Map

```mermaid
graph TB
    subgraph "Level 1: Foundation"
        NP[NumPy Fundamentals]
        PD[Pandas Essentials]
        EX[Data Exploration]
    end
    
    subgraph "Level 2: Data Preparation"
        CL[Data Cleaning]
        TR[Data Transformation]
        FE[Feature Engineering]
    end
    
    subgraph "Level 3: Model Fundamentals"
        RG[Regression Basics]
        CF[Classification Basics]
        EV[Model Evaluation]
    end
    
    subgraph "Level 4: Advanced"
        CLU[Clustering]
        DIM[Dimensionality Reduction]
        ENS[Ensemble Methods]
        TUNE[Hyperparameter Tuning]
    end
    
    subgraph "Level 5: Best Practices"
        PIPE[Pipelines]
        ETH[Ethics & Bias]
    end
    
    NP --> PD
    PD --> EX
    EX --> CL
    CL --> TR
    TR --> FE
    FE --> RG
    FE --> CF
    RG --> EV
    CF --> EV
    EV --> CLU
    EV --> DIM
    CLU --> ENS
    DIM --> ENS
    ENS --> TUNE
    TUNE --> PIPE
    PIPE --> ETH
```

## Validation Framework Architecture

```mermaid
classDiagram
    class KoanValidator {
        -notebook_id: str
        -koans: dict
        -results: dict
        +__init__(notebook_id)
        +koan(num, title, difficulty)
        +_mark_success(koan_num)
        +_mark_failure(koan_num, error)
        +_mark_error(koan_num, error)
    }
    
    class ProgressTracker {
        -progress_file: Path
        -data: dict
        +__init__(progress_file)
        +complete_koan(notebook_id, koan_num, score)
        +get_progress(notebook_id)
        +get_mastery_report()
        +display_progress()
        -_load_progress()
        -_save_progress()
        -_calc_mastery(notebooks)
    }
    
    class DataGenerator {
        +for_regression(n_samples, n_features, noise)
        +for_classification(n_samples, n_features, n_classes)
        +for_clustering(n_samples, n_features, n_clusters)
        +synthetic_tabular(rows, cols, types)
        +load_sklearn_dataset(name)
    }
    
    KoanValidator --> ProgressTracker : updates
    KoanValidator --> DataGenerator : uses
```

## Data Flow

```mermaid
graph LR
    A[Raw Concept] --> B[Koan Design]
    B --> C[TODO Exercise]
    C --> D[Learner Solution]
    D --> E[Validation]
    
    E --> F{Correct?}
    F -->|Yes| G[Update Progress]
    F -->|No| H[Show Feedback]
    
    G --> I[Calculate Mastery]
    I --> J[Save to JSON]
    
    H --> D
    
    style A fill:#e1f5e1
    style G fill:#c8e6c9
    style H fill:#ffcdd2
    style J fill:#b3e5fc
```

## Mastery Calculation

```mermaid
graph TB
    START[Completed Koans] --> GROUP[Group by Topic]
    GROUP --> CALC[Calculate % Complete]
    CALC --> WEIGHT[Apply Difficulty Weights]
    WEIGHT --> SCORE[Mastery Score 0-100%]
    
    SCORE --> L1{Score >= 90%}
    SCORE --> L2{Score >= 70%}
    SCORE --> L3{Score >= 50%}
    
    L1 -->|Yes| M1[🏆 Master]
    L2 -->|Yes| M2[⭐ Proficient]
    L3 -->|Yes| M3[📚 Learning]
    L3 -->|No| M4[🌱 Beginner]
    
    style M1 fill:#4caf50
    style M2 fill:#8bc34a
    style M3 