"""
Main agent definition for the RAG AI agent.
"""

import os
import sys
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TypedDict, Union
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from supabase import create_client, SupabaseException
from openai import OpenAI

from dotenv import load_dotenv
from pathlib import Path

# Try to import tiktoken, fall back to word count estimation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️ tiktoken not available, using word count estimation for tokens")

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import (
    KnowledgeBaseSearch,
    KnowledgeBaseSearchParams,
    KnowledgeBaseSearchResult,
)
from agent.prompts import RAG_SYSTEM_PROMPT, QUERY_PLANNER_PROMPT, ANSWER_WRITER_PROMPT

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Supabase-Konfiguration (Service-Role-Key!)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANNON_KEY")

# Lazy-init placeholder
global_supabase = None


def get_supabase_client():
    global global_supabase
    if global_supabase is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise SupabaseException("SUPABASE_URL und SUPABASE_SERVICE_ROLE_KEY müssen gesetzt sein")
        global_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return global_supabase


@dataclass
class AgenticRetrievalConfig:
    """
    Configuration for Agentic Retrieval system.
    """
    max_rounds: int = 2  # Optimized with metadata search - most questions solved in 1-2 rounds
    k_per_round: int = 15
    min_similarity: float = 0.55
    token_budget: int = 2000
    recency_halflife_days: int = 30
    min_sources: int = 2  # Reduced for faster early stopping
    min_sections: int = 3
    doc_type_preference: Optional[str] = None
    enable_filters: bool = True
    early_stopping_threshold: float = 0.75  # Stop if we find very good matches
    quality_over_speed: bool = False  # Toggle for comprehensive vs fast search

    @classmethod
    def from_env(cls) -> 'AgenticRetrievalConfig':
        """Create config from environment variables."""
        return cls(
            max_rounds=int(os.getenv('RAG_ROUNDS', '2')),
            k_per_round=int(os.getenv('K_PER_ROUND', '15')),
            min_similarity=float(os.getenv('MIN_SIM', '0.55')),
            token_budget=int(os.getenv('RAG_TOKEN_BUDGET', '2000')),
            recency_halflife_days=int(os.getenv('RECENCY_HALFLIFE_DAYS', '30')),
            doc_type_preference=os.getenv('DOC_TYPE_PREFERENCE'),
            enable_filters=os.getenv('RAG_ENABLE_FILTERS', 'true').lower() == 'true',
            early_stopping_threshold=float(os.getenv('RAG_EARLY_STOPPING_THRESHOLD', '0.75')),
            quality_over_speed=os.getenv('RAG_QUALITY_OVER_SPEED', 'false').lower() == 'true'
        )


class AgenticRetrievalOrchestrator:
    """
    Advanced multi-round retrieval orchestrator with planning, search, and synthesis.
    """
    
    def __init__(
        self, 
        kb_search: Optional[KnowledgeBaseSearch] = None,
        openai_client: Optional[OpenAI] = None,
        db_client = None  # Direct Supabase client (e.g., from get_sb_user())
    ):
        if db_client is not None:
            # Use the provided db_client for KnowledgeBaseSearch
            self.kb_search = kb_search or KnowledgeBaseSearch(db_client=db_client)
        else:
            self.kb_search = kb_search or KnowledgeBaseSearch()
        
        self.openai_client = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.db_client = db_client
        
    async def run(self, question: str, config: Optional[AgenticRetrievalConfig] = None) -> Dict[str, Any]:
        """
        Execute multi-round agentic retrieval and synthesis.
        
        Args:
            question: User's question
            config: Optional configuration override
            
        Returns:
            Dict with answer and source chunks
        """
        config = config or AgenticRetrievalConfig.from_env()
        
        print(f"\n🤖 === AGENTISCHES RETRIEVAL START ===")
        print(f"📋 Config: {config.max_rounds} Runden, {config.k_per_round} pro Runde, Token-Budget: {config.token_budget}")
        
        # Step 1: Query Planning
        sub_queries = await self._plan_queries(question)
        print(f"📝 Geplant: {len(sub_queries)} Sub-Queries")
        
        # Step 2: Multi-round search
        all_chunks = []
        unique_sections = set()
        unique_sources = set()
        
        for round_num, sub_query in enumerate(sub_queries[:config.max_rounds], 1):
            print(f"\n🔍 Runde {round_num}: {sub_query}")
            
            # Execute search with filters
            search_params = KnowledgeBaseSearchParams(
                query=sub_query,
                max_results=config.k_per_round,
                k_per_round=config.k_per_round,
                filters=self._build_filters(config) if config.enable_filters else None
            )
            
            # Standard content search
            round_results = await self.kb_search.search(search_params)
            round_chunks = self._convert_to_chunk_dicts(round_results)
            
            # Additional metadata search for cross-source discovery
            metadata_chunks = await self._search_document_metadata(sub_query)
            round_chunks.extend(metadata_chunks)
            
            print(f"   📝 {len(round_results)} Content-Treffer + {len(metadata_chunks)} Metadaten-Treffer")
            
            # Track coverage and quality
            round_max_score = 0.0
            for chunk in round_chunks:
                section_id = (chunk.get('url', ''), chunk.get('section_heading', ''))
                unique_sections.add(section_id)
                unique_sources.add(chunk.get('url', ''))
                round_max_score = max(round_max_score, chunk.get('similarity', 0.0))
                
            all_chunks.extend(round_chunks)
            
            print(f"   ✅ {len(round_chunks)} neue Chunks, gesamt: {len(all_chunks)}")
            print(f"   📊 Coverage: {len(unique_sources)} Quellen, {len(unique_sections)} Abschnitte")
            print(f"   🎯 Beste Similarity: {round_max_score:.3f}")
            
            # Smart Early Stopping for Cross-Source Retrieval
            # Track real document diversity (not just different sections of same doc)
            unique_doc_names = set()
            for chunk in all_chunks:
                url = chunk.get('url', '')
                # Extract document name (e.g., "trgs_611.pdf" from full path)
                doc_name = url.split('/')[-1] if '/' in url else url
                unique_doc_names.add(doc_name)
            
            sufficient_coverage = (len(unique_doc_names) >= config.min_sources and 
                                 len(unique_sections) >= config.min_sections)
            
            excellent_quality = round_max_score >= config.early_stopping_threshold
            
            # For Cross-Source questions: Ensure we execute at least half of planned queries
            min_rounds_for_cross_source = max(2, len(sub_queries) // 2)  # At least 50% of planned queries
            
            # Enhanced thematic diversity check for Cross-Source detection
            queries_so_far = sub_queries[:round_num]
            
            # Classify query themes more intelligently
            def classify_query_theme(query):
                query_upper = query.upper()
                if 'TRGS' in query_upper or 'NITRIT' in query_upper or 'GRENZWERT' in query_upper:
                    return 'REGULATIONS'  # Regulatory/compliance topics
                elif 'BOAT' in query_upper or 'SYNTH' in query_upper or any(oil_term in query_upper for oil_term in ['ÖL', 'OIL', 'MOTOR']):
                    return 'PRODUCT'  # Product-specific topics  
                else:
                    # Fallback to first two words
                    words = query.split()[:2]
                    return ' '.join(words).upper()
            
            executed_themes = set(classify_query_theme(q) for q in queries_so_far)
            all_planned_themes = set(classify_query_theme(q) for q in sub_queries)
            
            thematic_diversity = len(executed_themes) > 1  # Have we executed different themes?
            all_themes_covered = executed_themes >= all_planned_themes  # Have we covered all planned themes?
            
            print(f"   🎭 Themen-Analyse: Ausgeführt={executed_themes}, Geplant={all_planned_themes}, Abgedeckt={all_themes_covered}")
            
            # Enhanced Early Stopping Logic
            can_stop_early = False
            stop_reason = ""
            
            if config.quality_over_speed:
                # Quality mode: Never stop early, execute all rounds
                can_stop_early = False
                stop_reason = "Quality mode - continuing all rounds"
            elif round_num < min_rounds_for_cross_source:
                # Always execute minimum rounds for potential cross-source questions
                can_stop_early = False
                stop_reason = f"Executing minimum {min_rounds_for_cross_source} rounds for cross-source coverage"
            elif all_themes_covered and sufficient_coverage and excellent_quality:
                # Only stop if we have covered ALL planned themes AND have good quality/coverage
                can_stop_early = True
                stop_reason = f"All themes covered ({all_planned_themes}) with excellent quality (Score: {round_max_score:.3f})"
            elif sufficient_coverage and not thematic_diversity and round_num >= min_rounds_for_cross_source and all_themes_covered:
                # Stop if good coverage but all queries are about same topic AND we've covered all themes
                can_stop_early = True  
                stop_reason = f"Single-topic coverage sufficient after {round_num} rounds"
            
            if can_stop_early:
                print(f"   🚀 Early Stop: {stop_reason}")
                break
            elif stop_reason:
                print(f"   ⏳ Continue: {stop_reason}")
        
        # Step 3: Deduplicate and score
        final_chunks = self._deduplicate_and_score(all_chunks, config)
        print(f"📊 Nach Deduplizierung: {len(final_chunks)} finale Chunks")
        
        # Step 4: Context packing
        context_chunks, total_tokens = self._pack_context(final_chunks, config.token_budget)
        print(f"📦 Kontext gepackt: {len(context_chunks)} Chunks, ~{total_tokens} Tokens")
        
        # Step 5: Generate answer
        answer = await self._generate_answer(question, context_chunks)
        
        print(f"🤖 === AGENTISCHES RETRIEVAL ENDE ===\n")
        
        return {
            "answer": answer,
            "source_chunks": context_chunks,
            "total_rounds": min(len(sub_queries), config.max_rounds),
            "total_chunks_found": len(all_chunks),
            "unique_sources": len(unique_sources),
            "unique_sections": len(unique_sections),
            "context_tokens": total_tokens
        }
    
    async def _plan_queries(self, question: str) -> List[str]:
        """Generate sub-queries using LLM planning."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUERY_PLANNER_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            raw_response = response.choices[0].message.content
            print(f"🧠 Query Planner Response:\n{raw_response}")
            
            # Parse numbered list
            sub_queries = []
            for line in raw_response.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    query = re.sub(r'^\d+\.\s*', '', line).strip()
                    if query:
                        sub_queries.append(query)
            
            return sub_queries[:6]  # Limit to max 6 queries
            
        except Exception as e:
            print(f"❌ Query Planning Fehler: {e}")
            # Fallback: use original question
            return [question]
    
    def _build_filters(self, config: AgenticRetrievalConfig) -> Dict[str, Any]:
        """Build search filters based on configuration."""
        filters = {}
        
        if config.doc_type_preference:
            filters["doc_type"] = config.doc_type_preference
            
        # Note: min_confidence filter can be too restrictive, remove for now
        # if config.min_similarity > 0.0:
        #     filters["min_confidence"] = max(0.0, config.min_similarity - 0.1)
        
        # Note: Recency filter removed - knowledge doesn't expire by age in this system
        # one_year_ago = datetime.now() - timedelta(days=365) 
        # filters["created_after"] = one_year_ago.isoformat()
        
        return filters
        
    async def _search_document_metadata(self, sub_query: str) -> List[Dict[str, Any]]:
        """Search in document_metadata table for title matches."""
        from database.setup import SupabaseClient
        
        try:
            if self.db_client:
                # Use provided User client (RLS-enabled)
                db = self.db_client
            else:
                # Fallback to Service client
                supabase_client = SupabaseClient()
                db = supabase_client.client
            
            print(f"   🔍 Metadata-Suche: '{sub_query}'")
            
            # Smart search in document_metadata - try multiple strategies
            metadata_matches = []
            
            # Strategy 1: Full query match
            response1 = db.table("document_metadata").select("*").ilike("title", f"%{sub_query}%").execute()
            if response1.data:
                metadata_matches.extend(response1.data)
            
            # Strategy 2: Individual keyword matches (for "Wunsch BOAT SYNTH 2-T" -> ["BOAT", "SYNTH", "2-T"])
            keywords = [word for word in sub_query.split() if len(word) > 2 and word not in ["der", "die", "das", "und", "oder", "für"]]
            for keyword in keywords[:3]:  # Max 3 keywords to avoid too many hits
                response2 = db.table("document_metadata").select("*").ilike("title", f"%{keyword}%").execute()
                if response2.data:
                    for item in response2.data:
                        # Avoid duplicates
                        if not any(existing['doc_id'] == item['doc_id'] for existing in metadata_matches):
                            metadata_matches.append(item)
            
            print(f"   📋 Document Metadata Search: {len(metadata_matches)} Titel-Treffer für '{sub_query}' (Keywords: {keywords})")
            
            # Convert metadata matches to chunks by fetching the actual content
            chunks = []
            for doc_meta in metadata_matches:
                doc_title = doc_meta.get("title", "")
                source_url = doc_meta.get("source_url", "")
                doc_type = doc_meta.get("doc_type", "")
                
                print(f"   🎯 Metadata Match: '{doc_title}' -> {source_url}")
                
                # Robust URL matching: Try exact match first, then title-based fallback
                content_response = db.table("rag_pages").select("*").eq("url", source_url).limit(5).execute()
                
                # Fallback: If no chunks found with source_url, try matching by title
                if not content_response.data and doc_title:
                    print(f"   🔄 No chunks found for {source_url}, trying title-based match...")
                    # Try various title formats
                    title_variants = [
                        doc_title,  # Original title
                        doc_title.replace('"', ''),  # Remove quotes
                        doc_title.replace('"', '"').replace('"', '"'),  # Smart quotes
                        f'Info zu Öl "{doc_title.split('"')[1]}"' if '"' in doc_title else doc_title,  # Format variants
                    ]
                    
                    for variant in title_variants:
                        if variant != source_url:  # Don't retry the already failed source_url
                            print(f"   🔄 Trying title variant: '{variant}'")
                            fallback_response = db.table("rag_pages").select("*").eq("url", variant).limit(5).execute()
                            if fallback_response.data:
                                content_response = fallback_response
                                print(f"   ✅ Found chunks using title variant: '{variant}'")
                                source_url = variant  # Update source_url for consistent metadata
                                break
                
                if content_response.data:
                    for chunk_data in content_response.data:
                        chunk = {
                            "content": chunk_data.get("content", ""),
                            "url": source_url,
                            "similarity": 0.80,  # High score for metadata matches - but not too high to dominate
                            "metadata": chunk_data.get("metadata", {}),
                            "source_type": "metadata_match",
                            "page": chunk_data.get("page", 1),
                            "page_heading": chunk_data.get("page_heading") or "",
                            "section_heading": chunk_data.get("section_heading") or "",
                            "token_count": chunk_data.get("token_count", 0),
                            "confidence": chunk_data.get("confidence", 0.9),
                            "created_at": doc_meta.get("created_at", ""),
                            "file_modified_at": doc_meta.get("file_modified_at", ""),
                            "metadata_title_match": doc_title,  # Mark as metadata match
                            "doc_type": doc_type
                        }
                        chunks.append(chunk)
                        print(f"     📄 Chunk hinzugefügt: {chunk['content'][:100]}...")
                        
            print(f"   ✅ Metadata Search Ergebnis: {len(chunks)} Chunks aus {len(metadata_matches)} Dokumenten")
            return chunks
            
        except Exception as e:
            print(f"⚠️ Document Metadata Search Fehler: {e}")
            return []
    
    def _convert_to_chunk_dicts(self, results: List[KnowledgeBaseSearchResult]) -> List[Dict[str, Any]]:
        """Convert search results to chunk dictionaries."""
        chunks = []
        for result in results:
            chunk = {
                "content": result.content,
                "url": result.source,
                "similarity": result.similarity,
                "metadata": result.metadata,
                "source_type": result.source_type,
                # Extract additional fields from metadata
                "page": result.metadata.get("page", 1),
                "page_heading": result.metadata.get("page_heading") or "",
                "section_heading": result.metadata.get("section_heading") or "",
                "token_count": result.metadata.get("token_count", 0),
                "confidence": result.metadata.get("confidence", 0.8),
                "created_at": result.metadata.get("created_at", ""),
                "file_modified_at": result.metadata.get("file_modified_at", "")
            }
            chunks.append(chunk)
        return chunks
    
    def _deduplicate_and_score(self, chunks: List[Dict[str, Any]], config: AgenticRetrievalConfig) -> List[Dict[str, Any]]:
        """Deduplicate by (doc_id, section_heading) and calculate final scores."""
        seen_sections = {}
        
        for chunk in chunks:
            # Create deduplication key
            doc_id = chunk.get("url", "unknown")
            section = chunk.get("section_heading", "") or ""
            key = (doc_id, section)
            
            # Calculate final score
            similarity = chunk.get("similarity", 0.0)
            rerank_score = chunk.get("rerank_score", similarity)  # Fallback to similarity
            confidence = max(0.0, min(1.0, chunk.get("confidence", 0.8)))  # Clamp confidence
            recency_bonus = self._calculate_recency_bonus(chunk, config.recency_halflife_days)
            
            final_score = (0.55 * similarity + 
                          0.30 * rerank_score + 
                          0.10 * confidence + 
                          0.05 * recency_bonus)
            
            chunk["final_score"] = final_score
            
            # Keep best scoring chunk per section
            if key not in seen_sections or final_score > seen_sections[key]["final_score"]:
                seen_sections[key] = chunk
        
        # Sort by final score descending
        final_chunks = sorted(seen_sections.values(), key=lambda x: x["final_score"], reverse=True)
        
        print(f"📊 Top scoring chunks:")
        for i, chunk in enumerate(final_chunks[:5]):
            url = chunk.get("url", "Unknown")
            score = chunk["final_score"]
            section = (chunk.get("section_heading") or "No section")[:50]
            print(f"  [{i+1}] {url} | Score: {score:.3f} | {section}")
        
        return final_chunks
    
    def _calculate_recency_bonus(self, chunk: Dict[str, Any], halflife_days: int) -> float:
        """Calculate recency bonus (0.0 to 1.0) based on document age."""
        try:
            # Try to get modification date
            date_str = chunk.get("file_modified_at") or chunk.get("created_at")
            if not date_str:
                return 0.0
                
            if isinstance(date_str, str):
                doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                doc_date = date_str
                
            # Calculate age in days
            age_days = (datetime.now() - doc_date.replace(tzinfo=None)).days
            
            # Exponential decay: bonus = exp(-age_days / halflife_days)
            import math
            bonus = math.exp(-age_days / halflife_days)
            return min(1.0, max(0.0, bonus))
            
        except (ValueError, TypeError, AttributeError):
            return 0.0  # Default for unparseable dates
    
    def _pack_context(self, chunks: List[Dict[str, Any]], token_budget: int) -> tuple[List[Dict[str, Any]], int]:
        """Pack chunks into context respecting token budget."""
        selected_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk.get("content", ""))
            
            # Check if chunk fits in remaining budget
            if total_tokens + chunk_tokens <= token_budget:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Try to split large chunk if possible
                content = chunk.get("content", "")
                if chunk_tokens > token_budget // 2 and len(content) > 200:
                    # Split at sentence boundaries
                    sentences = re.split(r'[.!?]\s+', content)
                    partial_content = ""
                    partial_tokens = 0
                    
                    for sentence in sentences:
                        sentence_tokens = self._count_tokens(sentence + ". ")
                        if partial_tokens + sentence_tokens <= (token_budget - total_tokens):
                            partial_content += sentence + ". "
                            partial_tokens += sentence_tokens
                        else:
                            break
                    
                    if partial_content.strip():
                        split_chunk = chunk.copy()
                        split_chunk["content"] = partial_content.strip()
                        split_chunk["token_count"] = partial_tokens
                        selected_chunks.append(split_chunk)
                        total_tokens += partial_tokens
                
                break  # Stop adding chunks
        
        return selected_chunks, total_tokens
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or fallback to word estimation."""
        if not text:
            return 0
            
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model("gpt-4")
                return len(encoding.encode(text))
            except Exception:
                pass
        
        # Fallback: word count * 1.3
        return int(len(text.split()) * 1.3)
    
    async def _generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate final answer using context chunks."""
        # Build context from chunks
        context_sections = []
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("url", "Unknown")
            section = chunk.get("section_heading") or ""
            
            section_header = f"--- Abschnitt {i} ---"
            if section:
                section_header += f" ({section})"
            section_header += f"\nQuelle: {source}"
            
            context_sections.append(f"{section_header}\n\n{content}")
        
        full_context = "\n\n" + "\n\n".join(context_sections)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANSWER_WRITER_PROMPT + full_context},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            print(f"📝 Generated Answer Length: {len(answer)} characters")
            return answer
            
        except Exception as e:
            print(f"❌ Answer Generation Fehler: {e}")
            return "Es ist ein Fehler bei der Antwortgenerierung aufgetreten."


class AgentDeps(TypedDict, total=False):
    kb_search: KnowledgeBaseSearch


class RAGAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        kb_search: Optional[KnowledgeBaseSearch] = None,
        db_client = None,  # Direct Supabase client (e.g., from get_sb_user())
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as an argument or environment variable."
            )

        if db_client is not None:
            self.kb_search = kb_search or KnowledgeBaseSearch(owner_agent=self, db_client=db_client)
        else:
            self.kb_search = kb_search or KnowledgeBaseSearch(owner_agent=self)
        self.search_tool = Tool(self.kb_search.search)

        self.agent = Agent(
            f"openai:{self.model}",
            system_prompt=RAG_SYSTEM_PROMPT,
            tools=[self.search_tool],
        )

    async def query(
        self, question: str, max_results: int = 5, source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        print("\n=== [Agent.query] ===")
        print("Frage:", question)

        deps = AgentDeps(kb_search=self.kb_search)
        result = await self.agent.run(question, deps=deps)
        response = result.output

        kb_results = []
        for tool_call in getattr(result, "tool_calls", []):
            if tool_call.tool.name == "search":
                kb_results = tool_call.result or []

        print(f"🔍 Treffer aus Tool 'search': {len(kb_results)}")
        for i, res in enumerate(kb_results):
            sim = res.get("similarity", 0.0)
            snippet = res.get("content", "").replace("\n", " ")[:100]
            print(f"[{i+1}] Score: {sim:.3f} | {snippet}...")

        return {"response": response, "kb_results": kb_results}

    async def get_available_sources(self) -> List[str]:
        return await self.kb_search.get_available_sources()


# Funktion aus Klasse herausgelöst
def format_source_reference(metadata: Dict[str, Any], short: bool = False) -> str:
    """
    Erzeugt on-demand eine signierte URL für private Supabase-Dokumente oder Notizen.
    Ignoriert vorhandene (veraltete) signed_url-Einträge aus dem Upload.
    """
    filename = metadata.get("original_filename", "Unbekanntes Dokument")
    page = metadata.get("page") or "?"
    bucket = metadata.get("source_filter", "privatedocs")

    logging.debug(f"Erzeuge Signed URL für Datei: {filename} im Bucket: {bucket}")

    if short:
        return filename

    client = get_supabase_client()
    try:
        # Versuche verschiedene Ansätze für TXT-Dateien
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_extension == 'txt':
            # Für TXT-Dateien versuche mit transform-Optionen
            try:
                res = client.storage.from_(bucket).create_signed_url(
                    filename, 
                    3600, 
                    {
                        "download": True,
                        "transform": {
                            "format": "text",
                            "quality": 100
                        }
                    }
                )
            except Exception:
                # Fallback: Standard signed URL
                res = client.storage.from_(bucket).create_signed_url(filename, 3600)
        else:
            res = client.storage.from_(bucket).create_signed_url(filename, 3600)
            
        signed = res.get("signedURL")
        if not signed:
            logging.error(f"Keine signed URL in Response: {res}")
            return f"**Quelle:** {filename}, Seite {page} (kein Link verfügbar)"
    except Exception as e:
        logging.error(f"Fehler beim Erstellen der signierten URL: {e}")
        return f"**Quelle:** {filename}, Seite {page} (Fehler beim Link-Aufbau)"

    if bucket == "notes":
        return f"**Quelle:** Notiz: {filename}\n\n[Notiz anzeigen]({signed})"

    # Bestimme Dateityp basierend auf Dateiendung
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension == 'txt':
        # Für TXT-Dateien keine Seitenangabe und anderer Link-Text
        return f"**Quelle:** {filename}\n\n[Textdatei öffnen]({signed})"
    else:
        # Für PDF-Dateien mit Seitenangabe
        if page:
            signed += f"#page={page}"
        page_info = f"Seite {page}" if page else "ohne Seitenangabe"
        return f"**Quelle:** {filename}, {page_info}\n\n[PDF öffnen]({signed})"


# Singleton-Instanz für einfachen Import
agent = RAGAgent()
