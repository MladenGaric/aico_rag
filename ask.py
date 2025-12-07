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
    vs = Chroma(
        collection_name=collection,
        persist_directory=db_dir,
        embedding_function=embeddings,
    )

    retriever = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": 0.5,
        }
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ti si pomoćnik koji odgovara tačno i sažeto na osnovu dostavljenih konteksta. "
         "Ako nema dovoljno informacija, reci da nisi siguran. "
         ),
        ("human",
         "Pitanje: {input}\n\nKontekst:\n{context}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt) # llm + prompt
    rag = create_retrieval_chain(retriever, qa_chain) # baza + llm + prompt
    return rag

def main():
    parser = argparse.ArgumentParser(description="Postavi pitanje RAG sistemu.")
    parser.add_argument("--question", type=str, default="Who was Alexei Navalny?",nargs="+", help="Tekst pitanja.")
    parser.add_argument("--db_dir", type=str, default="local_chroma", help="Folder Chroma baze.")
    parser.add_argument("--collection", type=str, default="docs", help="Naziv kolekcije.")
    parser.add_argument("--k", type=int, default=4, help="Broj dokumenata za retrieval.")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY nije postavljen. Dodaj ga u okruženje ili .env.example")

    question = " ".join(args.question)
    rag_chain = build_rag_chain(args.db_dir, args.collection, k=args.k)
    result = rag_chain.invoke({"input": question})

    answer = result.get("answer")
    context = result.get("context", [])
    sources = []
    for d in context:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        if src not in sources: # deduplikacija (ubacuje samo nevidjene source)
            sources.append(src)

    print("\n=== ODGOVOR ===")
    print(answer.strip())
    if sources:
        print("\n=== IZVORI ===")
        for s in sources:
            print(f"- {s}")

if __name__ == "__main__":
    main()
