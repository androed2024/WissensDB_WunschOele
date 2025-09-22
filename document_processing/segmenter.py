"""
AgenticSegmenter for intelligent document chunking with heading detection and token-based boundaries.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from document_processing.tokens import count_tokens

logger = logging.getLogger(__name__)


class AgenticSegmenter:
    """
    Intelligent segmenter that detects headings and sections, then creates chunks 
    with token-based boundaries while respecting document structure.
    """
    
    def __init__(self, 
                 max_tokens: int = None,
                 soft_max_tokens: int = None, 
                 min_tokens: int = None,
                 overlap_tokens: int = None):
        """
        Initialize AgenticSegmenter with token limits.
        
        Args:
            max_tokens: Hard maximum token limit for chunks
            soft_max_tokens: Soft maximum - try not to exceed but allow if needed
            min_tokens: Minimum tokens for a chunk (merge smaller ones)
            overlap_tokens: Maximum overlap tokens between chunks
        """
        # Get parameters from environment variables with defaults
        self.max_tokens = max_tokens or int(os.getenv('RAG_MAX_TOKENS', '500'))
        self.soft_max_tokens = soft_max_tokens or int(os.getenv('RAG_SOFT_MAX_TOKENS', '650'))
        self.min_tokens = min_tokens or int(os.getenv('RAG_MIN_TOKENS', '120'))
        self.overlap_tokens = overlap_tokens or int(os.getenv('RAG_OVERLAP_TOKENS', '40'))
        
        logger.info(f"AgenticSegmenter initialized: max={self.max_tokens}, "
                   f"soft_max={self.soft_max_tokens}, min={self.min_tokens}, "
                   f"overlap={self.overlap_tokens}")
        
        # Regex patterns for heading detection
        self.heading_patterns = [
            # Numbered headings: "1.", "1.1", "2.3.4"
            re.compile(r'^(\d+(?:\.\d+)*\.?)\s+(.+)$', re.MULTILINE),
            # All caps headings (at least 3 chars, not too long)
            re.compile(r'^([A-ZÄÖÜ][A-ZÄÖÜ\s]{2,40})$', re.MULTILINE),
            # Markdown-style headings
            re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            # Underlined headings (next line is ===== or -----)
            re.compile(r'^(.+)\n([=\-]{3,})$', re.MULTILINE),
        ]
    
    def _detect_headings(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Detect headings in text using regex patterns.
        
        Args:
            text: Text to search for headings
            
        Returns:
            List of (start_pos, heading_text, pattern_type) tuples
        """
        headings = []
        
        for i, pattern in enumerate(self.heading_patterns):
            for match in pattern.finditer(text):
                start_pos = match.start()
                if i == 0:  # Numbered headings
                    heading_text = f"{match.group(1)} {match.group(2)}"
                    pattern_type = "numbered"
                elif i == 1:  # All caps
                    heading_text = match.group(1).strip()
                    pattern_type = "allcaps"
                elif i == 2:  # Markdown
                    heading_text = f"{match.group(1)} {match.group(2)}"
                    pattern_type = "markdown"
                elif i == 3:  # Underlined
                    heading_text = match.group(1).strip()
                    pattern_type = "underlined"
                
                headings.append((start_pos, heading_text, pattern_type))
        
        # Sort by position and remove duplicates (keep the first pattern match)
        headings.sort(key=lambda x: x[0])
        unique_headings = []
        last_pos = -1
        for pos, text, pattern in headings:
            if pos > last_pos + 10:  # Allow some spacing between headings
                unique_headings.append((pos, text, pattern))
                last_pos = pos
        
        logger.debug(f"Detected {len(unique_headings)} headings")
        return unique_headings
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using punctuation and line breaks.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, question marks
        # and line endings, but be careful with abbreviations
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?\n':
                # Check if this might be an abbreviation (short word before period)
                words = current.strip().split()
                if char == '.' and words and len(words[-1]) <= 3 and words[-1].isupper():
                    # Likely abbreviation, continue
                    continue
                
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        
        # Add remaining text
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if s]
    
    def _create_section_chunks(self, text: str, page_num: int, 
                              page_heading: Optional[str] = None,
                              section_heading: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create chunks from a section of text, respecting token limits.
        
        Args:
            text: Section text to chunk
            page_num: Page number
            page_heading: Heading for the page
            section_heading: Heading for the section
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            
            # If adding this sentence would exceed soft_max_tokens, finalize current chunk
            if (current_tokens + sentence_tokens > self.soft_max_tokens and 
                current_chunk and current_tokens >= self.min_tokens):
                
                # Check if we should add some overlap
                overlap_text = ""
                if chunks:  # Not the first chunk
                    overlap_sentences = sentences[max(0, len(sentences)-2):][:1]  # Last sentence as overlap
                    overlap_text = " ".join(overlap_sentences)
                    overlap_tokens = count_tokens(overlap_text)
                    if overlap_tokens <= self.overlap_tokens:
                        current_chunk = overlap_text + " " + current_chunk
                        current_tokens += overlap_tokens
                
                chunks.append({
                    "text": current_chunk,
                    "page": page_num,
                    "page_heading": page_heading,
                    "section_heading": section_heading,
                    "token_count": current_tokens
                })
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
                
                # Hard limit check - if we exceed max_tokens, we must split
                if current_tokens > self.max_tokens and len(current_chunk) > 200:
                    # Emergency split at word boundary
                    words = current_chunk.split()
                    split_point = len(words) // 2
                    
                    first_part = " ".join(words[:split_point])
                    second_part = " ".join(words[split_point:])
                    
                    chunks.append({
                        "text": first_part,
                        "page": page_num,
                        "page_heading": page_heading,
                        "section_heading": section_heading,
                        "token_count": count_tokens(first_part)
                    })
                    
                    current_chunk = second_part
                    current_tokens = count_tokens(second_part)
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk,
                "page": page_num,
                "page_heading": page_heading,
                "section_heading": section_heading,
                "token_count": current_tokens
            })
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge chunks that are smaller than min_tokens with their neighbors 
        (within the same section).
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If current chunk is too small and we have a next chunk in the same section
            if (current_chunk["token_count"] < self.min_tokens and 
                i + 1 < len(chunks) and 
                chunks[i + 1]["section_heading"] == current_chunk["section_heading"]):
                
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_text = current_chunk["text"] + " " + next_chunk["text"]
                merged_tokens = count_tokens(merged_text)
                
                # Only merge if the result doesn't exceed soft_max_tokens
                if merged_tokens <= self.soft_max_tokens:
                    merged_chunk = current_chunk.copy()
                    merged_chunk["text"] = merged_text
                    merged_chunk["token_count"] = merged_tokens
                    merged.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    logger.debug(f"Merged small chunk: {current_chunk['token_count']} + "
                               f"{next_chunk['token_count']} = {merged_tokens} tokens")
                    continue
            
            # If current chunk is too small and we have a previous chunk we can merge with
            if (current_chunk["token_count"] < self.min_tokens and 
                merged and 
                merged[-1]["section_heading"] == current_chunk["section_heading"]):
                
                prev_chunk = merged[-1]
                merged_text = prev_chunk["text"] + " " + current_chunk["text"]
                merged_tokens = count_tokens(merged_text)
                
                # Only merge if the result doesn't exceed soft_max_tokens
                if merged_tokens <= self.soft_max_tokens:
                    prev_chunk["text"] = merged_text
                    prev_chunk["token_count"] = merged_tokens
                    logger.debug(f"Merged small chunk backwards: {prev_chunk['token_count']} + "
                               f"{current_chunk['token_count']} = {merged_tokens} tokens")
                    i += 1
                    continue
            
            # No merging possible or necessary
            merged.append(current_chunk)
            i += 1
        
        return merged
    
    def segment_pages(self, pages: List[str]) -> List[Dict[str, Any]]:
        """
        Segment a list of pages into chunks with heading detection and token limits.
        
        Args:
            pages: List of page texts
            
        Returns:
            List of chunk dictionaries with keys:
            - text: Chunk text content
            - page: Page number (1-indexed)
            - page_heading: First heading found on the page (or None)
            - section_heading: Heading of the section this chunk belongs to (or None)
            - token_count: Number of tokens in the chunk
        """
        all_chunks = []
        stats = {
            "total_chunks": 0,
            "merged_small": 0,
            "split_large": 0,
            "pages_processed": 0
        }
        
        for page_idx, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue
            
            stats["pages_processed"] += 1
            
            # Detect headings on this page
            headings = self._detect_headings(page_text)
            
            # Determine page heading (first heading on the page)
            page_heading = headings[0][1] if headings else None
            
            if not headings:
                # No headings found, treat entire page as one section
                page_chunks = self._create_section_chunks(
                    page_text, page_idx, page_heading=page_heading, section_heading=None
                )
                all_chunks.extend(page_chunks)
                continue
            
            # Process sections between headings
            current_pos = 0
            
            for i, (heading_pos, heading_text, pattern_type) in enumerate(headings):
                # Add text before this heading (if any) to previous section
                if heading_pos > current_pos:
                    preceding_text = page_text[current_pos:heading_pos].strip()
                    if preceding_text:
                        # This text belongs to the previous section or no section
                        prev_section = headings[i-1][1] if i > 0 else None
                        section_chunks = self._create_section_chunks(
                            preceding_text, page_idx, 
                            page_heading=page_heading,
                            section_heading=prev_section
                        )
                        all_chunks.extend(section_chunks)
                
                # Determine the end of this section
                next_heading_pos = (headings[i+1][0] if i+1 < len(headings) 
                                  else len(page_text))
                
                # Extract section content (including the heading)
                section_start = heading_pos
                section_end = next_heading_pos
                section_text = page_text[section_start:section_end].strip()
                
                if section_text:
                    section_chunks = self._create_section_chunks(
                        section_text, page_idx,
                        page_heading=page_heading,
                        section_heading=heading_text
                    )
                    all_chunks.extend(section_chunks)
                
                current_pos = next_heading_pos
        
        # Merge small chunks
        initial_count = len(all_chunks)
        all_chunks = self._merge_small_chunks(all_chunks)
        stats["merged_small"] = initial_count - len(all_chunks)
        stats["total_chunks"] = len(all_chunks)
        
        # Calculate statistics
        token_counts = [chunk["token_count"] for chunk in all_chunks]
        if token_counts:
            stats["median_tokens"] = sorted(token_counts)[len(token_counts) // 2]
            stats["p90_tokens"] = sorted(token_counts)[int(len(token_counts) * 0.9)]
            stats["split_large"] = sum(1 for tc in token_counts if tc > self.soft_max_tokens)
        
        # Log statistics
        logger.info(f"AgenticSegmenter completed: {stats['total_chunks']} chunks from "
                   f"{stats['pages_processed']} pages")
        logger.info(f"Statistics: median_tokens={stats.get('median_tokens', 0)}, "
                   f"p90_tokens={stats.get('p90_tokens', 0)}, "
                   f"merged_small={stats['merged_small']}, "
                   f"split_large={stats['split_large']}")
        
        return all_chunks
