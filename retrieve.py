from typing import List
from embed_store import EmbedStore, Document

def retrieve_top_chunks(store: EmbedStore, query: str, k: int = 5) -> List[Document]:
    return store.search(query, k)
