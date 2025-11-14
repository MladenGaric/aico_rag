import os
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def build_rag_chain(db_dir: str, collection: str, k: int = 4):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        collection_name=collection,
        persist_directory=db_dir,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(
        search_type="mmr",  # ili "similarity"
        search_kwargs={"k": k}
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ti si pomoćnik koji odgovara tačno i sažeto na osnovu dostavljenih konteksta. "
         "Ako nema dovoljno informacija, reci da nisi siguran. "
         "Na kraju odgovora navedi kratku listu izvora (filename)."),
        ("human",
         "Pitanje: {input}\n\nKontekst:\n{context}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, qa_chain)
    return rag

def main():
    parser = argparse.ArgumentParser(description="Postavi pitanje RAG sistemu.")
    parser.add_argument("question", type=str, nargs="+", help="Tekst pitanja.")
    parser.add_argument("--db_dir", type=str, default="local_chroma", help="Folder Chroma baze.")
    parser.add_argument("--collection", type=str, default="docs", help="Naziv kolekcije.")
    parser.add_argument("--k", type=int, default=4, help="Broj dokumenata za retrieval.")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY nije postavljen. Dodaj ga u okruženje ili .env.")

    question = " ".join(args.question) if args.question else "Who was head of Nigeria in 1966 after coup?"
    rag_chain = build_rag_chain(args.db_dir, args.collection, k=args.k)
    result = rag_chain.invoke({"input": question})

    answer = result.get("answer") or result.get("output_text") or ""
    ctx = result.get("context", [])
    sources = []
    for d in ctx:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        if src not in sources:
            sources.append(src)

    print("\n=== ODGOVOR ===")
    print(answer.strip())
    if sources:
        print("\n=== IZVORI ===")
        for s in sources:
            print(f"- {s}")

if __name__ == "__main__":
    main()
