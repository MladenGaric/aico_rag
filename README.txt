aico_rag/
├── data/
│   └── docs/                      # Input .txt documents (Wikipedia texts)
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


1. My solution design was simple and modular. I first created a simple RAG solution with chromadb vectorestore. I provided it data
from Wiki and created a prompt that tells the LLM model what it should do based on what text. I used both MMR and cosine
similarity for search, but per default I made it MMR.
After that, I created a simple Knowledge Graph using pydentic and networkX, to extract information about the data regarding
triplets, having in mind Subject-relation-object schema, and I also wanted to show evidence on the made relation, along
the datasource it was extracted from.
After that, I created a simple ask_graph.py script in order to ask LLM anything, where he would gather data from KB.
On top of that, I combined the two, creating a hybrid approach utlizing both RAG and KG.

2. I used specific tech mostly because it is well-known tech-stack. Langchain, OpenAI, chromaDB and networkX are
all well known frameworks to work in when building such systems.

3. Yes, of course! I faced many challenged of which telemetry annoying stuff was the worst. When I finally tought I
found the way to scilence it, poof, there it is again. So that callenge i overcame with my inner strength and decided
to ignore it. Other than that, I faced challenges with various library versions, where it was hard to find the right
combination of libraries to work in. I checked in documentation, and finally found the combination to work with!
Also, I faced chellange with chromaDB, where I couldn't insert all of data at once, but instead I created a batch insertion method.


4. The KG improves on RAG system:
a) Explicit relationships
b) Graph paths reveal connections that RAG cannot infer
c) LLM gets ready-made factual triples instead of scanning raw text.


