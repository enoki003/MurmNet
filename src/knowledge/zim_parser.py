"""ZIM knowledge access utilities."""

import hashlib
import re
import threading
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from loguru import logger

try:
    from libzim.reader import Archive
    from libzim.search import Query, Searcher
    ZIM_AVAILABLE = True
except ImportError:
    logger.warning("libzim not available. ZIM parsing and search will be disabled.")
    Archive = None  # type: ignore
    Searcher = None  # type: ignore
    Query = None  # type: ignore
    ZIM_AVAILABLE = False

from src.config import config
from src.knowledge.embeddings import EmbeddingModel


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return cosine similarity between two vectors."""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


class DocumentChunk:
    """Represents a chunk of text from a document."""
    
    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, str],
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
    
    def __repr__(self) -> str:
        return f"DocumentChunk(id={self.chunk_id}, length={len(self.text)})"


class ZIMParser:
    """
    Parser for Wikipedia ZIM files.
    
    Extracts articles and splits them into chunks for vectorization.
    """
    
    def __init__(
        self,
        zim_path: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize ZIM parser.
        
        Args:
            zim_path: Path to ZIM file
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.zim_path = zim_path or config.knowledge_base.zim_file_path
        self.chunk_size = chunk_size or config.knowledge_base.chunk_size
        self.chunk_overlap = chunk_overlap or config.knowledge_base.chunk_overlap
        
        self.archive: Optional['Archive'] = None
        self.retrieval_top_k = config.knowledge_base.retrieval_top_k
        self.retrieval_threshold = config.knowledge_base.retrieval_score_threshold
        self.retrieval_debug = config.knowledge_base.retrieval_debug
        self._embedding_cache_size = config.knowledge_base.embedding_cache_size
    self._embedding_cache: Dict[str, np.ndarray] = {}
    self._cache_order: List[str] = []
    self._cache_lock = threading.Lock()
    self._embedding_model: Optional[EmbeddingModel] = None
        
        if not ZIM_AVAILABLE:
            logger.warning("ZIM support is not available")
            return
        
        if self.zim_path and self.zim_path.exists():
            self._open_archive()
    
    def _open_archive(self) -> None:
        """Open the ZIM archive."""
        if not ZIM_AVAILABLE:
            return
        
        try:
            self.archive = Archive(str(self.zim_path))
            logger.info(f"Opened ZIM archive: {self.zim_path}")
            logger.info(f"Archive contains {self.archive.all_entry_count} entries")
            logger.info(f"Article count: {self.archive.article_count}")
        except Exception as e:
            logger.error(f"Failed to open ZIM archive: {e}")
            self.archive = None

    def _get_embedding_model(self) -> Optional[EmbeddingModel]:
        """Lazily instantiate the embedding model used for similarity scoring."""
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            self._embedding_model = EmbeddingModel()
            if self.retrieval_debug:
                logger.debug(
                    "Initialized embedding model '%s' for ZIM retrieval",
                    self._embedding_model.model_name,
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(f"Failed to initialize embedding model: {exc}")
            self._embedding_model = None
        return self._embedding_model

    @staticmethod
    def _embedding_key(text: str) -> str:
        """Return stable hash for caching embeddings."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_cached_vector(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding vector if present."""
        with self._cache_lock:
            vector = self._embedding_cache.get(key)
            if vector is not None and key in self._cache_order:
                self._cache_order.remove(key)
                self._cache_order.append(key)
            return vector

    def _store_cached_vector(self, key: str, vector: np.ndarray) -> None:
        """Store embedding vector in cache with simple LRU eviction."""
        if self._embedding_cache_size <= 0:
            return
        with self._cache_lock:
            while (
                self._embedding_cache_size > 0
                and len(self._embedding_cache) >= self._embedding_cache_size
                and self._cache_order
            ):
                oldest = self._cache_order.pop(0)
                self._embedding_cache.pop(oldest, None)
            self._embedding_cache[key] = vector
            self._cache_order.append(key)

    def _get_embedding_vector(self, text: str) -> Optional[np.ndarray]:
        """Encode text into an embedding vector with caching."""
        model = self._get_embedding_model()
        if model is None:
            return None
        key = self._embedding_key(text)
        cached = self._get_cached_vector(key)
        if cached is not None:
            return cached
        embeddings = model.encode([text], show_progress=False)
        if not embeddings:
            return None
        vector = np.asarray(embeddings[0], dtype=np.float32)
        self._store_cached_vector(key, vector)
        return vector
    
    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML content and extract text.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            Cleaned text
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'meta', 'link']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _split_into_chunks(
        self,
        text: str,
        article_title: str,
        article_url: str,
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            article_title: Title of the article
            article_url: URL of the article
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split by sentences (simple approach)
        sentences = re.split(r'[。\.\n]+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk = DocumentChunk(
                        chunk_id=f"{article_url}#chunk{chunk_index}",
                        text=current_chunk.strip(),
                        metadata={
                            "title": article_title,
                            "url": article_url,
                            "chunk_index": str(chunk_index),
                            "source": "Wikipedia",
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last N characters as overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunk = DocumentChunk(
                chunk_id=f"{article_url}#chunk{chunk_index}",
                text=current_chunk.strip(),
                metadata={
                    "title": article_title,
                    "url": article_url,
                    "chunk_index": str(chunk_index),
                    "source": "Wikipedia",
                },
            )
            chunks.append(chunk)
        
        return chunks

    def retrieve_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        max_chunks_per_article: int = 2,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search ZIM archive and return relevant chunks."""
        if not query or not query.strip():
            return []
        if not ZIM_AVAILABLE or Searcher is None or Query is None:
            logger.debug("ZIM search unavailable; returning no chunks")
            return []
        if not self.archive:
            logger.debug("ZIM archive not loaded; returning no chunks")
            return []

        top_k = top_k or self.retrieval_top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self.retrieval_threshold
        )

        try:
            searcher = Searcher(self.archive)
            results = searcher.search(Query().set_query(query.strip()))
        except Exception as exc:
            logger.error(f"ZIM search failed: {exc}")
            return []

        if hasattr(results, "get_matches_estimated"):
            estimated = results.get_matches_estimated()
            if estimated == 0:
                if self.retrieval_debug:
                    logger.debug("No ZIM matches estimated for query '%s'", query)
                return []
        else:
            estimated = None

        query_vector = self._get_embedding_vector(query) if threshold > 0 else None
        if query_vector is None:
            threshold = 0.0

        retrieved: List[Tuple[DocumentChunk, float]] = []
        seen_chunk_ids = set()
        rank = 0

        while len(retrieved) < top_k:
            try:
                result = results.get_next()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug(f"ZIM search iteration failed: {exc}")
                break

            if not result:
                break

            try:
                entry = self.archive.get_entry_by_path(result.get_path())
            except Exception:
                continue

            if entry is None:
                continue
            if entry.is_redirect:
                entry = entry.get_redirect_entry()

            try:
                item = entry.get_item()
            except Exception:
                continue

            if getattr(item, "mimetype", None) != "text/html":
                continue

            try:
                html_content = bytes(item.content).decode("utf-8", errors="ignore")
            except Exception:
                continue

            text = self._clean_html(html_content)
            if len(text) < 50:
                continue

            chunks = self._split_into_chunks(
                text=text,
                article_title=entry.title,
                article_url=entry.path,
            )

            for chunk in chunks[:max_chunks_per_article]:
                if chunk.chunk_id in seen_chunk_ids:
                    continue

                similarity = 1.0
                if query_vector is not None:
                    chunk_vector = self._get_embedding_vector(chunk.text)
                    if chunk_vector is None:
                        continue
                    similarity = _cosine_similarity(query_vector, chunk_vector)
                    if similarity < threshold:
                        if self.retrieval_debug:
                            logger.debug(
                                "Filtered chunk %s (similarity %.3f < %.3f)",
                                chunk.chunk_id,
                                similarity,
                                threshold,
                            )
                        continue
                else:
                    similarity = max(0.0, 1.0 - (len(retrieved) / (top_k + 1)))

                chunk.metadata.update(
                    {
                        "retrieval_mode": "zim_search",
                        "retrieval_rank": str(rank),
                        "similarity": f"{similarity:.4f}",
                    }
                )

                retrieved.append((chunk, similarity))
                seen_chunk_ids.add(chunk.chunk_id)
                rank += 1

                if len(retrieved) >= top_k:
                    break

        if self.retrieval_debug:
            logger.debug(
                "ZIM retrieval produced %d chunks (threshold=%.2f, estimated=%s)",
                len(retrieved),
                threshold,
                estimated,
            )

        return retrieved
    
    def get_article_by_url(self, url: str) -> Optional[str]:
        """
        Get article content by URL.
        
        Args:
            url: Article URL (path)
            
        Returns:
            Article text or None if not found
        """
        if not self.archive:
            return None
        
        try:
            if not self.archive.has_entry_by_path(url):
                return None
            
            entry = self.archive.get_entry_by_path(url)
            if entry.is_redirect:
                entry = entry.get_redirect_entry()
            
            item = entry.get_item()
            html_content = bytes(item.content).decode('utf-8')
            
            return self._clean_html(html_content)
        except Exception as e:
            logger.error(f"Failed to get article {url}: {e}")
            return None
    
    def parse_all_articles(
        self,
        max_articles: Optional[int] = None,
    ) -> Generator[List[DocumentChunk], None, None]:
        """
        Parse all articles in the ZIM file and yield chunks.
        
        Args:
            max_articles: Maximum number of articles to process
            
        Yields:
            List of DocumentChunk objects for each article
        """
        if not self.archive:
            logger.error("No ZIM archive available")
            return
        
        processed = 0
        
        for entry_id in range(self.archive.all_entry_count):
            if max_articles and processed >= max_articles:
                break
            
            try:
                entry = self.archive._get_entry_by_id(entry_id)
                
                # Skip redirects
                if entry.is_redirect:
                    continue
                
                # Get item
                try:
                    item = entry.get_item()
                except:
                    # Not an article item (maybe metadata or image)
                    continue
                
                # Check if it's HTML content (article)
                if item.mimetype != 'text/html':
                    continue
                
                html_content = bytes(item.content).decode('utf-8')
                
                # Clean and extract text
                text = self._clean_html(html_content)
                
                if len(text) < 100:  # Skip very short articles
                    continue
                
                # Split into chunks
                chunks = self._split_into_chunks(
                    text=text,
                    article_title=entry.title,
                    article_url=entry.path,
                )
                
                if chunks:
                    yield chunks
                    processed += 1
                    
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} articles")
            
            except Exception as e:
                logger.debug(f"Skipping entry {entry_id}: {e}")
                continue
        
        logger.info(f"Finished processing {processed} articles")
    
    def search_articles(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search for articles matching a query using title-based search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of article metadata dictionaries
        """
        if not self.archive or not ZIM_AVAILABLE:
            return []
        
        # Simple title-based search since fulltext search may not be available
        results = []
        query_lower = query.lower()
        
        try:
            # Iterate through entries and match by title
            for entry_id in range(min(self.archive.all_entry_count, 10000)):
                if len(results) >= max_results:
                    break
                
                try:
                    entry = self.archive._get_entry_by_id(entry_id)
                    
                    # Skip redirects
                    if entry.is_redirect:
                        continue
                    
                    # Check if title matches query
                    if query_lower in entry.title.lower():
                        results.append({
                            "title": entry.title,
                            "url": entry.path,
                        })
                except:
                    continue
            
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def close(self) -> None:
        """Close the ZIM archive."""
        if self.archive:
            self.archive = None
            logger.info("Closed ZIM archive")
