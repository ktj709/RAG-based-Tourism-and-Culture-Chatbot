from typing import List
from embed_store import Document

def chunk_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """Split documents into smaller chunks"""
    chunks = []
    for doc in docs:
        text = doc.page_content
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(Document(
                page_content=chunk_text,
                metadata={**doc.metadata, 'chunk_start': start}
            ))
            start += chunk_size - chunk_overlap
    return chunks
