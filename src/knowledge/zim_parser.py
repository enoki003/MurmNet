"""
ZIM file parser for Wikipedia knowledge base.
Handles extraction and chunking of Wikipedia articles from ZIM format.
"""

import re
from pathlib import Path
from typing import Dict, Generator, List, Optional

from bs4 import BeautifulSoup
from loguru import logger

try:
    from libzim.reader import Archive
    ZIM_AVAILABLE = True
except ImportError:
    logger.warning("libzim not available. ZIM parsing will be disabled.")
    ZIM_AVAILABLE = False

from src.config import config


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
