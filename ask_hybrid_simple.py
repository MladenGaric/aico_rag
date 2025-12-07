import os
import argparse
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# RAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# KG
import networkx as nx
from rapidfuzz import process, fuzz
from langchain_core.pydantic_v1 import BaseModel, Field

DEF_DB_DIR = "local_chroma"
DEF_COLLECTION = "docs"
GRAPHML_PATH = Path("artifacts/kg.graphml")

def _truncate(txt: str, max_chars: int = 1200) -> str:
    return txt if len(txt) <= max_chars else txt[:max_chars] + "..."

def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def rag_topk_lc(
    question: str,
    db_dir: str,
    collection: str,
    k: int,
    score_threshold: float = 0.7
) -> List[Document]:

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = Chroma(
        collection_name=collection,
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    retriever = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
        },
    )

    docs: List[Document] = retriever.invoke(question)
    return docs


class KGQuery(BaseModel):
    entity: Optional[str] = Field(None, description="Glavni entitet u pitanju.")
    relation: Optional[str] = Field(None, description="Ako je pomenuta konkretna relacija (npr. 'born in').")
    limit: int = Field(5, description="Koliko ivica vratiti (neighbors).")

def load_graph() -> Optional[nx.MultiDiGraph]:
    if not GRAPHML_PATH.exists():
        return None

    return nx.read_graphml(GRAPHML_PATH)

def fuzzy_find_node(G: nx.Graph, name: str, score_cutoff=0.6) -> Optional[str]:
    candidates = {nid: (data.get("label") or nid) for nid, data in G.nodes(data=True)}
    candidates = {
        nid: lab
        for nid, lab in candidates.items()
        if str(lab).strip()
    }

    match = process.extractOne(name, candidates.values(), scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if not match:
        return None
    label = match[0]
    for nid, lab in candidates.items():
        if lab.lower() in label.lower() or lab.lower() == label.lower():
            return nid
    return None

def kg_neighbors(question: str, kg: nx.MultiDiGraph, llm: ChatOpenAI, limit: int) -> List[dict]:
    parser = llm.with_structured_output(KGQuery)

    plan = parser.invoke([
        ("system",
         "Extract the main entity (and optional relation) from the user's question. "
         "If multiple, pick the most central one. Return compact JSON."),
        ("user", question),
    ])

    if not plan.entity:
        return []

    nid = fuzzy_find_node(kg, name=plan.entity)
    if not nid:
        return []

    rel_filter = plan.relation.lower() if plan.relation else None
    out = []
    for u, v, key, data in kg.out_edges(nid, keys=True, data=True):
        rel = (data.get("relation") or "").lower()
        if rel_filter and rel != rel_filter:
            continue
        out.append({
            "subject": kg.nodes[u].get("label", u),
            "relation": rel or "related to",
            "object": kg.nodes[v].get("label", v),
            "source": data.get("source", ""),
            "evidence": (data.get("evidence", "") or "").replace("\n", " ")
        })
        if len(out) >= min(plan.limit, limit): # Opcionalno: Moze da se izbrise, ali onda vraca sve edges za nadjeni node
            break
    return out

def build_messages(question: str, rag_docs: List[Document], kg_facts: List[dict]) -> List[tuple]:
    rag_blocks, rag_sources = [], []
    for i, d in enumerate(rag_docs, 1):
        src = d.metadata.get("source") or d.metadata.get("path") or f"doc_{i}"
        rag_sources.append(src)
        rag_blocks.append(f"[Chunk {i} | {src}]\n{_truncate(d.page_content, 1000)}")
    rag_ctx = "\n\n".join(rag_blocks) if rag_blocks else "(no RAG context)"

    kg_lines, kg_sources = [], []
    for idx, f in enumerate(kg_facts, 1):
        kg_sources.append(f.get("source") or "")
        evid = f.get("evidence") or ""
        kg_lines.append(
            f"[KG Fact {idx}] "
            f"{f['subject']} --{f['relation']}--> {f['object']}  "
            f"[src: {f.get('source', '')}]  {('evidence: ' + evid) if evid else ''}"
        )
    kg_ctx = "\n".join(kg_lines) if kg_lines else "(no KG facts)"

    sys = (
        "You are a concise assistant. Use both sections if available.\n"
        "- Use KG facts for crisp relational answers.\n"
        "- Use RAG passages for descriptions/details.\n"
        "If missing info, say you are unsure.\n"
        )

    usr = f"Question: {question}\n\n=== RAG ===\n{rag_ctx}\n\n=== KG ===\n{kg_ctx}"

    return [("system", sys), ("user", usr)], _uniq(rag_sources), _uniq(kg_sources)

def main():
    ap = argparse.ArgumentParser(description="Simple Hybrid QA (RAG + KG neighbors).")
    ap.add_argument("--question", type=str, default= "Who is Alexei Navalny?",nargs="+", help="Pitanje.")
    ap.add_argument("--db_dir", default=DEF_DB_DIR)
    ap.add_argument("--collection", default=DEF_COLLECTION)
    ap.add_argument("--k", type=int, default=3, help="RAG top-k pasusa.")
    ap.add_argument("--kg_limit", type=int, default=3, help="Max KG cinjenica.")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY nije postavljen.")

    q = " ".join(args.question)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rag_docs = rag_topk_lc(q, args.db_dir, args.collection, k=args.k, score_threshold=0.15)

    kg = load_graph()
    kg_facts = kg_neighbors(q, kg, llm, limit=args.kg_limit)

    messages, rag_srcs, kg_srcs = build_messages(q, rag_docs, kg_facts)
    response = llm.invoke(messages)

    output = (response.content or "").strip()

    if rag_srcs:
        print("\n=== IZVORI (RAG) ===")
        for s in rag_srcs:
            print(f"- {s}")
    if kg_srcs:
        print("\n=== IZVORI (KG) ===")
        for s in kg_srcs:
            if s:
                print(f"- {s}")

    print("\n=== ODGOVOR (HYBRID SIMPLE) ===")
    print(output)


if __name__ == "__main__":
    main()