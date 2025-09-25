"""
Knowledge base search tool for the RAG AI agent.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.setup import SupabaseClient
from document_processing.embeddings import EmbeddingGenerator
from document_processing.reranker import CrossEncoderReranker


class KnowledgeBaseSearchParams(BaseModel):
    """
    Parameters for the knowledge base search tool.
    """

    query: str = Field(
        ...,
        description="The search query to find relevant information in the knowledge base",
    )
    max_results: int = Field(
        15, description="Maximum number of results to return (default: 15)"
    )
    source_filter: Optional[str] = Field(
        None, description="Optional filter to search only within a specific source"
    )
    k_per_round: int = Field(
        15, description="Number of candidates to retrieve per search round (default: 15)"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Advanced filters: doc_type, source, min_confidence, section_heading_contains, created_after"
    )


class KnowledgeBaseSearchResult(BaseModel):
    """
    Result from the knowledge base search.
    """

    content: str = Field(..., description="Content of the document chunk")
    source: str = Field(..., description="Source of the document chunk")
    source_type: str = Field(..., description="Type of source (e.g., 'pdf', 'txt')")
    similarity: float = Field(
        ..., description="Similarity score between the query and the document"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Additional metadata about the document"
    )


class KnowledgeBaseSearch:
    """
    Tool for searching the knowledge base using vector similarity.
    """

    def __init__(
        self,
        supabase_client: Optional[SupabaseClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        owner_agent: Optional[
            Any
        ] = None,  # Referenz zum Agenten, um Treffer dort abzulegen
        db_client = None,  # Direct Supabase client (e.g., from get_sb_user())
    ):
        """
        Initialize the knowledge base search tool.

        Args:
            supabase_client: SupabaseClient instance for database operations
            embedding_generator: EmbeddingGenerator instance for creating embeddings
            owner_agent: Optional reference to the RAGAgent to store last_match results
            db_client: Direct Supabase client (takes priority over supabase_client)
        """
        if db_client is not None:
            # Use direct client (e.g., from get_sb_user())
            self.db_client = db_client
            self.supabase_client = None  # Mark as using direct client
        else:
            # Use SupabaseClient wrapper
            self.supabase_client = supabase_client or SupabaseClient()
            self.db_client = self.supabase_client.client
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.owner_agent = owner_agent
        self.reranker = CrossEncoderReranker()

    async def search(
        self, params: KnowledgeBaseSearchParams
    ) -> List[KnowledgeBaseSearchResult]:
        """
        Search the knowledge base for relevant information.

        Args:
            params: Search parameters

        Returns:
            List of search results
        """
        print("\n---[RAG Retrieval]---")
        print(f"🔍 [LLM REFORMULIERT] Such-Query: {params.query}")

        # Generate embedding for the query
        query_embedding = self.embedding_generator.embed_text(params.query)

        # Prepare filter metadata - combine legacy source_filter and new filters
        filter_metadata = {}
        
        # Legacy source_filter support
        if params.source_filter:
            filter_metadata["source"] = params.source_filter
            
        # New advanced filters
        if params.filters:
            # Apply Supabase-compatible filters
            if "source" in params.filters:
                filter_metadata["source"] = params.filters["source"]
            if "doc_type" in params.filters:
                filter_metadata["doc_type"] = params.filters["doc_type"]

        # Use k_per_round for candidate count
        candidate_count = max(params.k_per_round, params.max_results)
        
        if self.supabase_client:
            # Use SupabaseClient methods
            vector_results = self.supabase_client.search_documents(
                query_embedding=query_embedding,
                match_count=candidate_count,
                filter_metadata=filter_metadata,
            )
            keyword_results = self.supabase_client.keyword_search_documents(
                params.query,
                match_count=candidate_count,
                filter_metadata=filter_metadata,
            )
        else:
            # Use direct client for search operations
            # Implement the same logic as SupabaseClient methods
            
            # Vector search using match_rag_pages RPC function
            try:
                match_threshold = float(os.getenv("RAG_MIN_SIM", "0.55"))
                
                rpc_params = {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": candidate_count,
                }

                if filter_metadata:
                    rpc_params["filter"] = filter_metadata

                result = self.db_client.rpc("match_rag_pages", rpc_params).execute()
                
                if result.data:
                    # Zusätzliche lokale Filterung nach Similarity
                    vector_results = [r for r in result.data if r.get("similarity", 0) >= match_threshold]
                    print(f"\n🔍 Top {len(vector_results)} RAG-Matches (von {len(result.data)} gefiltert):")
                    for r in vector_results:
                        score = r.get("similarity", 0.0)
                        preview = r["content"][:120].replace("\n", " ")
                        print(f"  • Score: {score:.3f} → {preview}...")
                else:
                    print("⚠️ Keine Dokument-Treffer für die Anfrage gefunden.")
                    vector_results = []
            except Exception as e:
                print("❌ Fehler bei Supabase-RPC:", str(e))
                vector_results = []

            # Keyword search
            try:
                qb = self.db_client.table("rag_pages").select(
                    "id,url,chunk_number,content,metadata"
                )
                if filter_metadata:
                    for key, value in filter_metadata.items():
                        qb = qb.contains("metadata", {key: value})
                qb = qb.ilike("content", f"%{params.query}%").limit(candidate_count)
                result = qb.execute()
                keyword_results = result.data or []
            except Exception as e:
                print("❌ Fehler bei Keyword-Suche:", e)
                keyword_results = []
        # Kombiniere Ergebnisse mit verbesserter Filterung
        # Keyword-Suche nur als Fallback, wenn Vector-Suche leer ist
        if vector_results:
            all_results = vector_results + [
                r for r in keyword_results if r not in vector_results
            ]
        else:
            # Fallback: Nur Keyword-Suche verwenden
            all_results = keyword_results
            print("⚠️ Keine Vector-Treffer - verwende Keyword-Suche als Fallback")
            
            # Additional Fallback: Search in document metadata if no content results
            if not all_results:
                print("🔍 Versuche Metadaten-Suche als letzten Fallback...")
                metadata_results = await self._search_document_metadata_fallback(params.query, candidate_count)
                all_results.extend(metadata_results)
                if metadata_results:
                    print(f"✅ {len(metadata_results)} Metadaten-Treffer gefunden")
        
        # Python-seitige erweiterte Filterung
        if params.filters:
            all_results = self._apply_python_filters(all_results, params.filters)
        
        # Score-basierte Filterung nach Merge
        MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.55"))
        filtered_results = [r for r in all_results if r.get("similarity", 0.0) >= MIN_SIM]
        
        # Falls zu wenige Treffer, lockere den Threshold etwas
        if len(filtered_results) == 0 and all_results:
            fallback_threshold = MIN_SIM * 0.8  # 20% niedriger
            filtered_results = [r for r in all_results if r.get("similarity", 0.0) >= fallback_threshold]
            if filtered_results:
                print(f"⚠️ Threshold auf {fallback_threshold:.2f} reduziert - {len(filtered_results)} Treffer")
                    
        results = filtered_results[: params.max_results]

        # Source Expansion: Add more chunks from same documents for better context  
        expanded_results = await self._expand_source_context(results, params.max_results)
        results = expanded_results

        print("\n📊 Reranker-Scores:")
        for i, r in enumerate(results[: params.max_results]):
            sim = r.get("similarity", -1)
            rerank = r.get("rerank_score", -1)
            snippet = r.get("content", "")[:120].replace("\n", " ")
            print(f"[{i+1}] sim={sim:.3f} | rerank={rerank:.3f} → {snippet}...")

        print(f"🔎 similarity_search → {len(results)} Treffer")
        for i, match in enumerate(results):
            sim = match.get("similarity", -1)
            url = match.get("url", "N/A")
            meta = match.get("metadata", {})
            title = meta.get("title") or meta.get("original_filename", "Unbekannt")
            content_snippet = match.get("content", "")[:200].replace("\n", " ")
            print(f"[{i+1}] {title} (Score: {sim:.3f}) | URL: {url}")
            print(f"     Textauszug: {content_snippet}...")

        # Prompt Context Debug
        context_snippets = [match.get("content", "")[:500] for match in results]
        full_context = "\n---\n".join(context_snippets)
        print("\n📋 Prompt Context (erster Teil):\n", full_context[:1000])

        # Convert results to KnowledgeBaseSearchResult objects
        search_results = []
        for result in results:
            search_results.append(
                KnowledgeBaseSearchResult(
                    content=result.get("content", ""),
                    source=result.get("metadata", {}).get("source", "Unknown"),
                    source_type=result.get("metadata", {}).get(
                        "source_type", "Unknown"
                    ),
                    similarity=float(result.get("similarity", 0.0)),
                    metadata=result.get("metadata", {}),
                )
            )

        # Optional: speichere Treffer im Agenten für UI-Anzeige (akkumulierend)
        if self.owner_agent is not None:
            if not hasattr(self.owner_agent, 'last_match') or self.owner_agent.last_match is None:
                self.owner_agent.last_match = []
                print("🔄 Initialisiere leeren last_match Akkumulator")
            
            # Akkumuliere alle Tool-Call Ergebnisse statt zu überschreiben
            previous_count = len(self.owner_agent.last_match)
            self.owner_agent.last_match.extend(results)
            print(f"🔄 Akkumulierte Treffer: {len(results)} neue + {previous_count} vorherige = {len(self.owner_agent.last_match)} gesamt")
            
            # Debug: Zeige was akkumuliert wurde
            for i, result in enumerate(results):
                url = result.get('url', 'Unknown')
                sim = result.get('similarity', 0.0)
                print(f"  └─ [{i+1}] {url} (Score: {sim:.3f})")

        return search_results

    def _apply_python_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply Python-side filters that cannot be handled by Supabase.
        
        Args:
            results: List of search results to filter
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered list of results
        """
        filtered_results = results.copy()
        
        # min_confidence filter
        if "min_confidence" in filters:
            min_conf = float(filters["min_confidence"])
            filtered_results = [r for r in filtered_results 
                              if r.get("confidence", 0.0) >= min_conf]
        
        # section_heading_contains filter  
        if "section_heading_contains" in filters:
            search_term = filters["section_heading_contains"].lower()
            filtered_results = [r for r in filtered_results 
                              if search_term in (r.get("section_heading") or "").lower()]
        
        # created_after filter (for recency)
        if "created_after" in filters:
            try:
                if isinstance(filters["created_after"], str):
                    cutoff_date = datetime.fromisoformat(filters["created_after"])
                else:
                    cutoff_date = filters["created_after"]
                    
                filtered_results = [r for r in filtered_results 
                                  if self._get_document_date(r) >= cutoff_date]
            except (ValueError, TypeError):
                print(f"⚠️ Ungültiges created_after Format: {filters['created_after']}")
        
        return filtered_results
        
    async def _search_document_metadata_fallback(self, query: str, max_results: int = 15) -> List[Dict[str, Any]]:
        """
        Search in document_metadata table for title matches as final fallback.
        Returns results in the same format as regular search.
        Improved version with better relevance scoring and specificity.
        """
        try:
            print(f"   🔍 Fallback-Metadaten-Suche: '{query}'")
            
            metadata_matches = []
            
            # Strategy 1: Full query match (highest priority)
            response1 = self.db_client.table("document_metadata").select("*").ilike("title", f"%{query}%").execute()
            if response1.data:
                for item in response1.data:
                    item['_match_score'] = 1.0  # Highest score for complete matches
                    item['_match_type'] = 'complete_query'
                metadata_matches.extend(response1.data)
                print(f"   ✅ Strategy 1: {len(response1.data)} complete query matches")
            
            # Only proceed with partial matching if no complete matches found
            if not metadata_matches:
                print("   🔄 No complete matches, trying product name extraction...")
                
                # Strategy 2: Extract and search for complete product names
                # Look for quoted product names or specific patterns like "Product XYZ"
                product_patterns = []
                
                # Pattern 1: Extract quoted product names
                import re
                quoted_matches = re.findall(r'"([^"]+)"', query)
                product_patterns.extend(quoted_matches)
                
                # Pattern 2: Detect product-like patterns (CAPS + numbers/hyphens)
                product_match = re.search(r'\b([A-Z]+(?:\s+[A-Z0-9-]+)*)\b', query)
                if product_match:
                    product_patterns.append(product_match.group(1))
                
                # Search for complete product patterns
                for pattern in product_patterns:
                    if pattern and len(pattern.strip()) > 3:
                        response2 = self.db_client.table("document_metadata").select("*").ilike("title", f"%{pattern}%").execute()
                        if response2.data:
                            for item in response2.data:
                                if not any(existing['doc_id'] == item['doc_id'] for existing in metadata_matches):
                                    item['_match_score'] = 0.9  # High score for product name matches
                                    item['_match_type'] = 'product_name'
                                    metadata_matches.append(item)
                            print(f"   ✅ Strategy 2: {len(response2.data)} product name matches for '{pattern}'")
            
            # Strategy 3: Multi-keyword combination (only if still no matches)
            if not metadata_matches:
                print("   🔄 No product matches, trying keyword combinations...")
                keywords = [word for word in query.split() if len(word) > 2 and word not in ["der", "die", "das", "und", "oder", "für", "technische", "Eigenschaften", "Vorteile", "Leistungsmerkmale"]]
                
                if len(keywords) >= 2:
                    # Try combinations of 2+ keywords
                    for i in range(len(keywords)):
                        for j in range(i+1, len(keywords)):
                            combo_pattern = f"%{keywords[i]}%{keywords[j]}%"
                            response3 = self.db_client.table("document_metadata").select("*").ilike("title", combo_pattern).execute()
                            if response3.data:
                                for item in response3.data:
                                    if not any(existing['doc_id'] == item['doc_id'] for existing in metadata_matches):
                                        item['_match_score'] = 0.7  # Medium score for keyword combinations
                                        item['_match_type'] = 'keyword_combo'
                                        metadata_matches.append(item)
                                print(f"   ✅ Strategy 3: {len(response3.data)} combo matches for '{keywords[i]}' + '{keywords[j]}'")
            
            # Strategy 4: Individual keywords (only as last resort and with lower scores)
            if not metadata_matches:
                print("   ⚠️ Last resort: trying individual keywords...")
                keywords = [word for word in query.split() if len(word) > 3 and word not in ["der", "die", "das", "und", "oder", "für", "technische", "Eigenschaften", "Vorteile", "Leistungsmerkmale", "Daten"]]
                
                for keyword in keywords[:2]:  # Max 2 keywords and only longer ones
                    response4 = self.db_client.table("document_metadata").select("*").ilike("title", f"%{keyword}%").execute()
                    if response4.data:
                        for item in response4.data:
                            if not any(existing['doc_id'] == item['doc_id'] for existing in metadata_matches):
                                # Calculate relevance based on keyword importance
                                score = 0.4 if len(keyword) > 4 else 0.3  # Longer keywords get higher scores
                                item['_match_score'] = score
                                item['_match_type'] = 'single_keyword'
                                metadata_matches.append(item)
                        print(f"   ⚠️ Strategy 4: {len(response4.data)} single keyword matches for '{keyword}' (score: {score})")
            
            # Sort by match score (highest first)
            metadata_matches.sort(key=lambda x: x.get('_match_score', 0), reverse=True)
            
            print(f"   📋 Metadata Fallback: {len(metadata_matches)} Titel-Treffer (sorted by relevance)")
            
            # Convert metadata matches to search result format
            results = []
            for doc_meta in metadata_matches[:max_results]:  # Limit results
                doc_title = doc_meta.get("title", "")
                source_url = doc_meta.get("source_url", "")
                doc_type = doc_meta.get("doc_type", "")
                match_score = doc_meta.get('_match_score', 0.5)
                match_type = doc_meta.get('_match_type', 'unknown')
                
                print(f"   🎯 Metadata Match: '{doc_title}' -> {source_url} (score: {match_score:.2f}, type: {match_type})")
                
                # Fetch actual content chunks from this document
                content_response = self.db_client.table("rag_pages").select("*").eq("url", source_url).limit(5).execute()
                
                # Fallback: If no chunks found with source_url, try matching by title
                if not content_response.data and doc_title:
                    print(f"   🔄 No chunks found for {source_url}, trying title-based match...")
                    # Try various title formats
                    title_variants = [
                        doc_title,  # Original title
                        doc_title.replace('"', ''),  # Remove quotes
                        doc_title.replace('"', '"').replace('"', '"'),  # Smart quotes
                        f'Info zu "{doc_title.split('"')[1]}"' if '"' in doc_title else doc_title,  # Format variants
                    ]
                    
                    for variant in title_variants:
                        if variant != source_url:  # Don't retry the already failed source_url
                            print(f"   🔄 Trying title variant: '{variant}'")
                            fallback_response = self.db_client.table("rag_pages").select("*").eq("url", variant).limit(5).execute()
                            if fallback_response.data:
                                content_response = fallback_response
                                print(f"   ✅ Found chunks using title variant: '{variant}'")
                                source_url = variant  # Update source_url for consistent metadata
                                break
                
                if content_response.data:
                    for chunk_data in content_response.data:
                        # Convert to standard search result format with improved scoring
                        result = {
                            "content": chunk_data.get("content", ""),
                            "url": source_url,
                            "similarity": match_score,  # Use calculated match score instead of fixed 0.80
                            "metadata": chunk_data.get("metadata", {}),
                            "source_type": "metadata_fallback",
                            "page": chunk_data.get("page", 1),
                            "page_heading": chunk_data.get("page_heading", ""),
                            "section_heading": chunk_data.get("section_heading", ""),
                            "token_count": chunk_data.get("token_count", 0),
                            "confidence": match_score,  # Use match score as confidence
                            "created_at": doc_meta.get("created_at", ""),
                            "file_modified_at": doc_meta.get("file_modified_at", ""),
                            "metadata_title_match": doc_title,  # Mark as metadata match
                            "metadata_match_type": match_type,  # Add match type for debugging
                            "doc_type": doc_type
                        }
                        results.append(result)
                        
            print(f"   ✅ Metadata Fallback Ergebnis: {len(results)} Chunks aus {len(metadata_matches)} Dokumenten")
            return results
            
        except Exception as e:
            print(f"⚠️ Metadata Fallback Fehler: {e}")
            return []
    
    def _get_document_date(self, result: Dict[str, Any]) -> datetime:
        """
        Extract document creation/modification date from result metadata.
        
        Args:
            result: Search result dictionary
            
        Returns:
            Document date or very old date as fallback
        """
        metadata = result.get("metadata", {})
        
        # Try various date fields
        for date_field in ["file_modified_at", "created_at", "fs_mtime"]:
            date_value = metadata.get(date_field)
            if date_value:
                try:
                    if isinstance(date_value, str):
                        return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    elif isinstance(date_value, datetime):
                        return date_value
                except (ValueError, TypeError):
                    continue
        
        # Fallback: very old date
        return datetime(1900, 1, 1)

    async def get_available_sources(self) -> List[str]:
        """
        Get a list of all available sources in the knowledge base.

        Returns:
            List of source identifiers
        """
        if self.supabase_client:
            return self.supabase_client.get_all_document_sources()
        else:
            # Direct client fallback
            result = self.db_client.table("rag_pages").select("url").execute()
            urls = set(item["url"] for item in result.data if result.data)
            return list(urls)
    
    async def _expand_source_context(self, results: List[Dict[str, Any]], max_total: int) -> List[Dict[str, Any]]:
        """
        Expand context by adding more chunks from the same sources.
        This helps with fragmented information in PDFs/documents.
        """
        if not results:
            return results
            
        print(f"🔍 Source-Expansion: Erweitere Kontext für {len(results)} Treffer...")
        
        # Group results by source URL
        sources_with_good_matches = {}
        for result in results:
            url = result.get("url", "")
            if url and result.get("similarity", 0) >= 0.55:  # Expand good matches (aligned with min_similarity)
                if url not in sources_with_good_matches:
                    sources_with_good_matches[url] = []
                sources_with_good_matches[url].append(result)
        
        expanded_results = list(results)  # Start with original results
        
        # For each source with good matches, fetch additional chunks
        for source_url, source_results in sources_with_good_matches.items():
            if len(expanded_results) >= max_total:
                break
                
            try:
                # Get all chunks from this document
                response = self.db_client.table("rag_pages").select("*").eq("url", source_url).order("chunk_number").limit(8).execute()
                
                if response.data:
                    print(f"   🔄 Erweitere {source_url}: +{len(response.data)} Chunks aus Dokument")
                    
                    for chunk_data in response.data:
                        # Skip if already in results
                        chunk_id = chunk_data.get("id")
                        if any(r.get("id") == chunk_id for r in expanded_results):
                            continue
                        
                        # Add with slightly lower score to maintain ranking
                        expanded_chunk = {
                            "content": chunk_data.get("content", ""),
                            "url": source_url,
                            "similarity": 0.55,  # Context chunks get decent score
                            "metadata": chunk_data.get("metadata", {}),
                            "source_type": "context_expansion",
                            "page": chunk_data.get("page", 1),
                            "page_heading": chunk_data.get("page_heading", ""),
                            "section_heading": chunk_data.get("section_heading", ""),
                            "token_count": chunk_data.get("token_count", 0),
                            "confidence": 0.6,
                            "id": chunk_id,
                            "chunk_number": chunk_data.get("chunk_number", 0)
                        }
                        
                        expanded_results.append(expanded_chunk)
                        
                        if len(expanded_results) >= max_total:
                            break
                            
            except Exception as e:
                print(f"   ⚠️ Source-Expansion Fehler für {source_url}: {e}")
        
        # Sort by similarity, but keep context chunks together
        expanded_results.sort(key=lambda x: (x.get("similarity", 0), -x.get("chunk_number", 999)), reverse=True)
        
        print(f"   ✅ Source-Expansion: {len(results)} → {len(expanded_results)} Chunks")
        return expanded_results[:max_total]
