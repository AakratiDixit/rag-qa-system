def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Chunk text into smaller pieces
    
    Args:
        text: Input text
        chunk_size: Number of characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        
        start += (chunk_size - overlap)
    
    return chunks