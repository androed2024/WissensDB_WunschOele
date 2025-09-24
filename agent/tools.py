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
            # Use direct client - fallback to basic search
            # Note: This is a simplified implementation for RLS compatibility
            vector_results = []
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
        """
        try:
            print(f"   🔍 Fallback-Metadaten-Suche: '{query}'")
            
            metadata_matches = []
            
            # Strategy 1: Full query match
            response1 = self.db_client.table("document_metadata").select("*").ilike("title", f"%{query}%").execute()
            if response1.data:
                metadata_matches.extend(response1.data)
            
            # Strategy 2: Individual keyword matches (for "Wunsch BOAT SYNTH 2-T" -> ["BOAT", "SYNTH", "2-T"])
            keywords = [word for word in query.split() if len(word) > 2 and word not in ["der", "die", "das", "und", "oder", "für", "Wunsch"]]
            for keyword in keywords[:3]:  # Max 3 keywords to avoid too many hits
                response2 = self.db_client.table("document_metadata").select("*").ilike("title", f"%{keyword}%").execute()
                if response2.data:
                    for item in response2.data:
                        # Avoid duplicates
                        if not any(existing['doc_id'] == item['doc_id'] for existing in metadata_matches):
                            metadata_matches.append(item)
            
            print(f"   📋 Metadata Fallback: {len(metadata_matches)} Titel-Treffer (Keywords: {keywords})")
            
            # Convert metadata matches to search result format
            results = []
            for doc_meta in metadata_matches[:max_results]:  # Limit results
                doc_title = doc_meta.get("title", "")
                source_url = doc_meta.get("source_url", "")
                doc_type = doc_meta.get("doc_type", "")
                
                print(f"   🎯 Metadata Match: '{doc_title}' -> {source_url}")
                
                # Fetch actual content chunks from this document
                content_response = self.db_client.table("rag_pages").select("*").eq("url", source_url).limit(5).execute()
                
                if content_response.data:
                    for chunk_data in content_response.data:
                        # Convert to standard search result format
                        result = {
                            "content": chunk_data.get("content", ""),
                            "url": source_url,
                            "similarity": 0.80,  # Good score for title matches - consistent with agentic
                            "metadata": chunk_data.get("metadata", {}),
                            "source_type": "metadata_fallback",
                            "page": chunk_data.get("page", 1),
                            "page_heading": chunk_data.get("page_heading", ""),
                            "section_heading": chunk_data.get("section_heading", ""),
                            "token_count": chunk_data.get("token_count", 0),
                            "confidence": chunk_data.get("confidence", 0.8),
                            "created_at": doc_meta.get("created_at", ""),
                            "file_modified_at": doc_meta.get("file_modified_at", ""),
                            "metadata_title_match": doc_title,  # Mark as metadata match
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
