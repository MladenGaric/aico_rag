from pathlib import Path
from typing import Literal, Optional, List
from dotenv import load_dotenv

import networkx as nx
from rapidfuzz import process, fuzz

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

GRAPHML = Path("artifacts/kg.graphml")

class KGQuery(BaseModel):
    kind: Literal["neighbors", "relation", "shortest_path"] = Field(..., description="Vrsta upita nad grafom.")
    entity: Optional[str] = Field(None, description="Glavni entitet (za neighbors/relation).")
    relation: Optional[str] = Field(None, description="Ako je poznata ciljna relacija (npr. 'born in').")
    entity_b: Optional[str] = Field(None, description="Drugi entitet (za shortest_path).")
    limit: int = Field(15, description="Koliko rezultata vratiti (podrazumevano 5).")

def fuzzy_find_node(G: nx.Graph, name: str, score_cutoff=0.9) -> Optional[str]:
    candidates = {nid: (data.get("label") or nid) for nid, data in G.nodes(data=True)}
    candidates = {
        nid: lab
        for nid, lab in candidates.items()
        if str(lab).strip()  # ovo izbacuje "" i "   "
    }

    match = process.extractOne(name, candidates.values(), scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if not match:
        return None
    label = match[0]
    for nid, lab in candidates.items():
        if lab.lower() in label.lower() or lab.lower() == label.lower():
            return nid
    return None

def plan_query(question: str, llm: ChatOpenAI) -> KGQuery:
    system = (
        "Map the user question to a simple knowledge-graph query.\n"
        "Choose one: 'neighbors', 'relation', or 'shortest_path'.\n"
        "If the question is 'Who/What is X' prefer 'neighbors' with limit≈10.\n"
        "If it asks for a specific predicate (e.g., born in), prefer 'relation'.\n"
        "If it asks how X is connected to Y, use 'shortest_path'.\n"
        "Return only the JSON object."
    )
    parser = llm.with_structured_output(KGQuery)
    return parser.invoke([("system", system), ("user", question)])

def execute_query(G: nx.MultiDiGraph, q: KGQuery) -> List[dict]:
    out = []

    if q.kind in ("neighbors", "relation"):
        if not q.entity:
            return out
        nid = fuzzy_find_node(G, q.entity)  # robustno mapiranje
        if not nid:
            return out

        for s, o, r, data in G.out_edges(nid, keys=True, data=True):
            rel = data.get("relation", "")
            if q.kind == "relation" and q.relation:
                if rel != q.relation.lower():
                    continue
            out.append({
                "subject": G.nodes[s].get("label", s),
                "relation": rel,
                "object": G.nodes[o].get("label", o),
                "source": data.get("source", ""),
                "evidence": data.get("evidence", "")
            })
            if len(out) >= q.limit:
                break

    elif q.kind == "shortest_path":
        if not (q.entity and q.entity_b):
            return out
        a = fuzzy_find_node(G, q.entity)
        b = fuzzy_find_node(G, q.entity_b)
        if not (a and b):
            return out
        try:
            path = nx.shortest_path(G.to_undirected(), a, b)
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                data = None
                if G.has_edge(u, v):
                    data = list(G.get_edge_data(u, v).values())[0]
                elif G.has_edge(v, u):
                    data = list(G.get_edge_data(v, u).values())[0]
                out.append({
                    "subject": G.nodes[u].get("label", u),
                    "relation": (data or {}).get("relation", "related to"),
                    "object": G.nodes[v].get("label", v),
                    "source": (data or {}).get("source", ""),
                    "evidence": (data or {}).get("evidence", "")
                })
        except nx.NetworkXNoPath:
            return out

    return out

def main():
    load_dotenv()
    if not GRAPHML.exists():
        raise RuntimeError("Nema grafa. Pokreni: kg_extract.py pa kg_build.py")

    MG = nx.read_graphml(GRAPHML)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    question = input("Unesi pitanje za KG: ").strip()
    # question = 'Who was Joseph Garba?'
    plan = plan_query(question, llm)
    results = execute_query(MG, plan)

    print("\n=== ODGOVOR (KG) ===")
    if not results:
        print("Nisam pronašao rezultat u grafu.")
        return

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['subject']}  --{r['relation']}-->  {r['object']}")
        if r.get("source"):
            print(f"   [source] {r['source']}")
        if r.get("evidence"):
            print(f"   [evidence] {r['evidence'][:200]}")

if __name__ == "__main__":
    main()
