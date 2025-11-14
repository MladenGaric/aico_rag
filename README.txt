aico_rag/
├── data/
│   └── documents/                 # Input .txt documents (Wikipedia texts)
├── local_chroma/                  # Vector database for RAG (automatically generated)
├── artifacts/
│   ├── kg_triples.jsonl           # Extracted KG triples (subject, relation, object)
│   ├── kg_nodes.csv               # List of graph nodes
│   ├── kg_edges.csv               # List of graph edges
│   └── kg.graphml                 # GraphML file for visualization (Gephi, Cytoscape)
├── ingest.py                      # Ingest documents into ChromaDB (vector store)
├── ask.py                         # Standard RAG query script
├── ask_graph.py                   # Knowledge Graph query script
├── ask_hybrid_simple.py           # Simple hybrid RAG + KG question answering
├── kg_extract.py                  # Extract KG triples from documents
├── kg_build.py                    # Build the Knowledge Graph from extracted triples
└── requirements.txt               # Python dependencies for the project




[0] Project setup
    ├─ Create virtual env  ->  python -m venv .venv
    ├─ Activate it         ->  .venv\Scripts\activate
    ├─ Install deps        ->  pip install -r requirements.txt
    └─ Add .env.example with       ->  OPENAI_API_KEY=sk-xxxx

        ↓

[1] Prepare data
    └─ Put all .txt Wikipedia files into:
       aico_rag/data/documents/

        ↓

[2] Build RAG index (vector store)
    └─ Run:
       python ingest.py --reset
       → creates/updates local_chroma/ with embeddings

        ↓

[3] Extract Knowledge Graph triples
    └─ Run:
       python kg_extract.py
       → writes artifacts/kg_triples.jsonl

        ↓

[4] Build Knowledge Graph
    └─ Run:
       python kg_build.py
       → creates:
          - artifacts/kg_nodes.csv
          - artifacts/kg_edges.csv
          - artifacts/kg.graphml

        ↓

[5] Ask questions (three options)
    ├─ Pure RAG:
    │    python ask.py "Your question..."
    │
    ├─ Pure KG (graph questions):
    │    python ask_graph.py
    │    → then type question in the terminal
    │
    └─ Hybrid RAG + KG (simple):
         python ask_hybrid_simple.py "Your question..." --k 5 --kg_limit 6


