import os
import json
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("data/documents")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "kg_triples.jsonl"

class Triple(BaseModel):
    subject: str = Field(..., description="Glavni entitet (kratak naziv).")
    relation: str = Field(..., description="Veza/predikat, malim slovima (npr. 'is a', 'located in', 'born in').")
    object: str = Field(..., description="Drugi entitet ili literal.")
    evidence: str = Field(..., description="Kratak citat (<=200 char) koji potkrepljuje trojku.")
    source: str = Field(..., description="Naziv fajla iz kog je trojka izvedena.")

class TripleList(BaseModel):
    triples: List[Triple] = Field(default_factory=list)

def load_documents() -> List[tuple[str, str]]:
    pairs = []
    for p in DOCS_DIR.rglob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = p.read_text(encoding="latin-1", errors="ignore")
        pairs.append((p.name, txt))
    if not pairs:
        raise RuntimeError(f"Nema .txt fajlova u {DOCS_DIR}")
    return pairs

def chunk_text(text: str, chunk_size=1200, overlap=150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nije postavljen.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extractor = llm.with_structured_output(TripleList)

    docs = load_documents()
    total = 0

    if OUT_JSONL.exists():
        OUT_JSONL.unlink()

    with OUT_JSONL.open("a", encoding="utf-8") as fw:
        for fname, text in docs:
            chunks = chunk_text(text)
            for ch in chunks:
                system = (
                    "Extract concise factual triples from the user text. "
                    "Prefer canonical entity names; use lowercase for simple relations. "
                    "Return only triples you can support with a short evidence quote."
                )
                user = f"Source file: {fname}\n\nText:\n{ch}"

                result: TripleList = extractor.invoke(
                    [("system", system), ("user", user)]
                )

                for t in result.triples:
                    t.source = fname
                    fw.write(json.dumps(t.dict(), ensure_ascii=False) + "\n")
                    total += 1

    print(f"[KG-EXTRACT] Triplets: {total:,} -> {OUT_JSONL}")

if __name__ == "__main__":
    main()
