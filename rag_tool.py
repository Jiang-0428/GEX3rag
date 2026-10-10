from rag import get_vector_store

def retrieve_policy_docs(query: str, top_k: int =4) -> list[str]:
    db = get_vector_store()
    docs = db.similarity_search(query, k=top_k)
    results = []
    for doc in docs:
        piece = f"[policy_id:{doc.metadata['policy_id']}, topic:{doc.metadata['topic']}]\n{doc.page_content}"
        results.append(piece)
    return results

if __name__ == "__main__":
    res = retrieve_policy_docs("What exact phone number should I call for the university IT Service Desk?")
    for r in res:
        print(r)
        print("-"*60)
