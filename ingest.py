import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMADB_TELEMETRY"] = "False"

from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import shutil

DEFAULT_DOCS_DIR = Path("data/documents")
DEFAULT_DB_DIR = Path("local_chroma")
DEFAULT_COLLECTION = "docs"

def load_txt_documents(docs_dir: Path) -> list[Document]:
    docs = []
    for p in docs_dir.rglob("*.txt"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = p.read_text(encoding="latin-1", errors="ignore")
        meta = {
            "source": str(p.relative_to(docs_dir)),
            "title": p.stem,
            "path": str(p.resolve()),
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def batched(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def main():
    parser = argparse.ArgumentParser(description="Ingest .txt dokumenata u lokalni Chroma vektor-store.")
    parser.add_argument("--docs_dir", type=str, default=str(DEFAULT_DOCS_DIR), help="Folder sa .txt dokumentima.")
    parser.add_argument("--db_dir", type=str, default=str(DEFAULT_DB_DIR), help="Folder za Chroma bazu.")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Naziv kolekcije u Chromi.")
    parser.add_argument("--reset", action="store_true", help="Obriši postojeću bazu pre ingest-a.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Veličina chunk-a.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Preklapanje chunk-ova.")
    parser.add_argument("--batch_size", type=int, default=2000, help="Max broj dokumenata po upisu (mora < Chroma max).")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY nije postavljen. Dodaj ga u okruženje ili .env.")

    docs_dir = Path(args.docs_dir)
    db_dir = Path(args.db_dir)

    if args.reset and db_dir.exists():
        shutil.rmtree(db_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Ne postoji folder sa dokumentima: {docs_dir}")

    raw_docs = load_txt_documents(docs_dir)
    if not raw_docs:
        raise RuntimeError(f"Nema .txt fajlova u {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[INGEST] Ukupno chunkova: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client_settings = Settings(
        is_persistent=True,
        anonymized_telemetry=False,
    )
    vs = Chroma(
        collection_name=args.collection,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    inserted = 0
    for idx, batch in enumerate(batched(chunks, args.batch_size), start=1):
        vs.add_documents(batch)
        inserted += len(batch)
        print(f"[INGEST] Batch {idx}: +{len(batch)} (ukupno {inserted})")

    count = vs._collection.count()
    print(f"[INGEST] GOTOVO. Kolekcija='{args.collection}', count={count}, db_dir='{db_dir}'")

if __name__ == "__main__":
    main()
