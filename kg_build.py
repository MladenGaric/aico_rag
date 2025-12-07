import json
from pathlib import Path
import networkx as nx
import pandas as pd

IN_JSONL = Path("artifacts/kg_triples.jsonl")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True, parents=True)

NODES_CSV = OUT_DIR / "kg_nodes.csv"
EDGES_CSV = OUT_DIR / "kg_edges.csv"
GRAPHML = OUT_DIR / "kg.graphml"

def norm(s: str) -> str:
    return " ".join(s.strip().split()).casefold()
# "straße".lower()      # "straße"
# "straße".casefold()   # "strasse"

def main():
    if not IN_JSONL.exists():
        raise RuntimeError(f"Nije pronađen fajl: {IN_JSONL}. Pokreni kg_extract.py prvo.")

    G = nx.MultiDiGraph()
    # DiGraph: A --> B <> B --> A
    # Multi: Allows for multiple edges between same nodes

    nodes_seen = {}
    edges = []

    with IN_JSONL.open("r", encoding="utf-8") as fr:
        for line in fr:
            t = json.loads(line)
            s_raw, r, o_raw = t["subject"], t["relation"], t["object"]
            ev, src = t.get("evidence", ""), t.get("source", "")

            s = norm(s_raw)
            o = norm(o_raw)
            r = norm(r)

            if s not in nodes_seen:
                nodes_seen[s] = {"label": s_raw}
            if o not in nodes_seen:
                nodes_seen[o] = {"label": o_raw}

            G.add_node(s, **nodes_seen[s])
            G.add_node(o, **nodes_seen[o])

            # ivica ima: relation, evidence, source, raw_subj/obj
            G.add_edge(s, o, relation=r, evidence=ev, source=src,
                       subj_label=s_raw, obj_label=o_raw)

            edges.append({
                "subject": s, "subject_label": s_raw,
                "relation": r,
                "object": o, "object_label": o_raw,
                "evidence": ev, "source": src
            })

    nodes_df = pd.DataFrame([{"node": nid, **data} for nid, data in G.nodes(data=True)])
    edges_df = pd.DataFrame(edges)

    nodes_df.to_csv(NODES_CSV, index=False, encoding="utf-8")
    edges_df.to_csv(EDGES_CSV, index=False, encoding="utf-8")

    nx.write_graphml(G, GRAPHML)

    print(f"[KG-BUILD] Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
    print(f"[KG-BUILD] Saved: {NODES_CSV.name}, {EDGES_CSV.name}, {GRAPHML.name} u {OUT_DIR}")

if __name__ == "__main__":
    main()
