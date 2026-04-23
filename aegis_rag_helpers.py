"""
Project Aegis: Advanced RAG System - Helper Classes
Complete implementation of all RAG components
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import tiktoken


class MarkdownSemanticChunker:
    """
    Advanced chunking that preserves:
    1. Header hierarchy
    2. Table integrity
    3. Semantic boundaries
    4. Token overlap between chunks
    """
    
    def __init__(self, max_tokens=500, overlap_percentage=0.15):
        self.max_tokens = max_tokens
        self.overlap_tokens = int(max_tokens * overlap_percentage)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def extract_headers(self, text: str) -> Dict[str, str]:
        """Extract current header context (h1, h2, h3)"""
        headers = {"h1": "", "h2": "", "h3": ""}
        lines = text.split('\n')
        
        for line in lines:
            if line.startswith('# '):
                headers['h1'] = line.replace('# ', '').strip()
                headers['h2'] = ""
                headers['h3'] = ""
            elif line.startswith('## '):
                headers['h2'] = line.replace('## ', '').strip()
                headers['h3'] = ""
            elif line.startswith('### '):
                headers['h3'] = line.replace('### ', '').strip()
        
        return headers
    
    def is_table(self, text: str) -> bool:
        """Detect if text contains a Markdown table"""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        has_separator = any('|' in line and '-' in line for line in lines[:3])
        has_pipes = sum(1 for line in lines if '|' in line) >= 2
        
        return has_separator and has_pipes
    
    def extract_table_with_headers(self, table_text: str) -> List[str]:
        """Split large tables by row, preserving column headers"""
        lines = table_text.strip().split('\n')
        if len(lines) < 3:
            return [table_text]
        
        header_row = lines[0]
        separator = lines[1]
        data_rows = lines[2:]
        
        if self.count_tokens(table_text) <= self.max_tokens:
            return [table_text]
        
        chunks = []
        for row in data_rows:
            chunk = f"{header_row}\n{separator}\n{row}"
            chunks.append(chunk)
        
        return chunks
    
    def split_by_headers(self, text: str) -> List[Dict]:
        """Split document by header hierarchy (supports both markdown and plain text)"""
        sections = []
        current_section = ""
        current_headers = {"h1": "", "h2": "", "h3": ""}
        
        lines = text.split('\n')
        
        for line in lines:
            # Check for markdown headers
            if line.startswith('#'):
                if current_section.strip():
                    sections.append({
                        "text": current_section.strip(),
                        "headers": current_headers.copy()
                    })
                
                if line.startswith('# '):
                    current_headers['h1'] = line.replace('# ', '').strip()
                    current_headers['h2'] = ""
                    current_headers['h3'] = ""
                elif line.startswith('## '):
                    current_headers['h2'] = line.replace('## ', '').strip()
                    current_headers['h3'] = ""
                elif line.startswith('### '):
                    current_headers['h3'] = line.replace('### ', '').strip()
                
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append({
                "text": current_section.strip(),
                "headers": current_headers.copy()
            })
        
        # If no markdown headers found, treat entire text as one section
        if len(sections) == 0 or (len(sections) == 1 and not any(sections[0]['headers'].values())):
            sections = [{
                "text": text.strip(),
                "headers": {"h1": "", "h2": "", "h3": ""}
            }]
        
        return sections
    
    def chunk_with_overlap(self, text: str, headers: Dict) -> List[Dict]:
        """Chunk text with token overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "headers": headers.copy(),
                "token_count": len(chunk_tokens)
            })
            
            start = end - self.overlap_tokens
        
        return chunks
    
    def chunk_document(self, text: str) -> List[Dict]:
        """Main chunking pipeline"""
        all_chunks = []
        sections = self.split_by_headers(text)
        
        for section in sections:
            section_text = section['text']
            headers = section['headers']
            
            if self.is_table(section_text):
                table_chunks = self.extract_table_with_headers(section_text)
                for table_chunk in table_chunks:
                    all_chunks.append({
                        "text": table_chunk,
                        "headers": headers,
                        "is_table": True,
                        "token_count": self.count_tokens(table_chunk)
                    })
            else:
                if self.count_tokens(section_text) > self.max_tokens:
                    chunks = self.chunk_with_overlap(section_text, headers)
                    all_chunks.extend(chunks)
                else:
                    all_chunks.append({
                        "text": section_text,
                        "headers": headers,
                        "is_table": False,
                        "token_count": self.count_tokens(section_text)
                    })
        
        return all_chunks


class MetadataExtractor:
    """Extract structured metadata from documents using LLM"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def extract_document_metadata(self, filename: str, full_text: str) -> Dict:
        """Extract document-level metadata using LLM"""
        
        prompt = f"""Analyze this corporate policy document and extract metadata.

Document filename: {filename}
First 1000 characters:
{full_text[:1000]}

Extract and return ONLY a JSON object with these fields:
{{
  "document_id": "short unique ID (e.g., TRV-POL-2005)",
  "policy_category": "one of: Travel, HR, IT, Finance, Legal, Operations, General",
  "policy_owner": "department or team name",
  "effective_date": "YYYY-MM-DD format or null",
  "document_title": "full title of the policy"
}}

Return ONLY valid JSON, no other text."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            metadata_str = response.choices[0].message.content.strip()
            metadata_str = re.sub(r'^```json\s*', '', metadata_str)
            metadata_str = re.sub(r'\s*```$', '', metadata_str)
            
            metadata = json.loads(metadata_str)
            return metadata
        
        except Exception as e:
            print(f"⚠️ Metadata extraction failed: {e}")
            return {
                "document_id": filename.replace('.md', '').replace(' ', '-'),
                "policy_category": "General",
                "policy_owner": "Unknown",
                "effective_date": None,
                "document_title": filename
            }
    
    def enrich_chunk_metadata(self, chunk: Dict, doc_metadata: Dict) -> Dict:
        """Combine chunk-level and document-level metadata"""
        return {
            "chunk_text": chunk['text'],
            "metadata": {
                **doc_metadata,
                "h1_header": chunk['headers'].get('h1', ''),
                "h2_header": chunk['headers'].get('h2', ''),
                "h3_header": chunk['headers'].get('h3', ''),
                "is_table": chunk.get('is_table', False),
                "token_count": chunk.get('token_count', 0),
                "ingestion_timestamp": datetime.now().isoformat()
            }
        }


class EmbeddingPipeline:
    """Generate embeddings and upsert to Pinecone"""
    
    def __init__(self, openai_client, pinecone_client, index_name, embedding_model, dimension):
        self.openai_client = openai_client
        self.pc = pinecone_client
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index = None
    
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        from pinecone import ServerlessSpec
        
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
        print(f"✅ Connected to index: {self.index_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def batch_embed(self, texts: List[str], batch_size=100) -> List[List[float]]:
        """Generate embeddings in batches"""
        from tqdm.auto import tqdm
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            time.sleep(0.1)
        
        return all_embeddings
    
    def upsert_chunks(self, enriched_chunks: List[Dict], batch_size=100):
        """Upsert chunks to Pinecone with embeddings"""
        from tqdm.auto import tqdm
        
        if not self.index:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        texts = [chunk['chunk_text'] for chunk in enriched_chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.batch_embed(texts)
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(enriched_chunks, embeddings)):
            vector_id = f"{chunk['metadata']['document_id']}_chunk_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    **chunk['metadata'],
                    "text": chunk['chunk_text']
                }
            })
        
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting"):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
            time.sleep(0.1)
        
        print(f"✅ Upserted {len(vectors)} vectors to Pinecone")
        return len(vectors)


class QueryTransformer:
    """Transform user queries using Multi-Query Expansion and HyDE"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate multiple variations of the query"""
        
        prompt = f"""You are a query expansion expert for corporate policy search.

Original query: "{query}"

Generate {num_variations} different ways to ask this same question, using varied terminology and phrasing that might appear in corporate policy documents.

Return ONLY the variations, one per line, without numbering or explanation."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        variations = response.choices[0].message.content.strip().split('\n')
        variations = [v.strip() for v in variations if v.strip()]
        
        all_queries = [query] + variations
        return all_queries[:num_variations + 1]
    
    def generate_hyde(self, query: str) -> str:
        """Generate hypothetical answer (HyDE)"""
        
        prompt = f"""You are writing a corporate policy document.

Question: "{query}"

Write a hypothetical policy excerpt that would answer this question. Use formal corporate language, include specific details, numbers, and procedures. Write 2-3 sentences.

Do NOT say you don't have information. Write as if you ARE the policy document."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()


class MetadataFilter:
    """Apply intelligent metadata filtering"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def detect_category(self, query: str) -> Optional[str]:
        """Detect policy category from query using LLM"""
        
        prompt = f"""Classify this query into ONE policy category.

Query: "{query}"

Categories: Travel, HR, IT, Finance, Legal, Operations, General

Return ONLY the category name, nothing else. If uncertain, return "General"."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        category = response.choices[0].message.content.strip()
        
        valid_categories = ["Travel", "HR", "IT", "Finance", "Legal", "Operations", "General"]
        return category if category in valid_categories else None
    
    def build_pre_filter(self, query: str) -> Dict:
        """Build Pinecone metadata filter"""
        category = self.detect_category(query)
        
        if category and category != "General":
            return {"policy_category": {"$eq": category}}
        
        return {}
    
    def post_filter_by_date(self, results: List[Dict]) -> List[Dict]:
        """Keep only most recent version of each policy"""
        
        by_doc = defaultdict(list)
        for result in results:
            doc_id = result['metadata'].get('document_id', 'unknown')
            by_doc[doc_id].append(result)
        
        filtered = []
        for doc_id, chunks in by_doc.items():
            sorted_chunks = sorted(
                chunks,
                key=lambda x: x['metadata'].get('effective_date') or '1900-01-01',
                reverse=True
            )
            filtered.extend(sorted_chunks[:3])
        
        return filtered


class Reranker:
    """Rerank retrieved chunks using Cohere ReRank"""
    
    def __init__(self, cohere_client):
        self.client = cohere_client
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        
        if not documents:
            return []
        
        texts = [doc['metadata']['text'] for doc in documents]
        
        rerank_response = self.client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=top_k
        )
        
        reranked = []
        for result in rerank_response.results:
            doc = documents[result.index].copy()
            doc['rerank_score'] = result.relevance_score
            reranked.append(doc)
        
        return reranked


class AdvancedRetriever:
    """Complete retrieval pipeline combining all techniques"""
    
    def __init__(self, openai_client, cohere_client, pinecone_index, embedding_model):
        self.openai_client = openai_client
        self.index = pinecone_index
        self.embedding_model = embedding_model
        
        self.query_transformer = QueryTransformer(openai_client)
        self.metadata_filter = MetadataFilter(openai_client)
        self.reranker = Reranker(cohere_client)
    
    def embed_query(self, query: str) -> List[float]:
        """Generate query embedding"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str, top_k: int = 5, use_hyde: bool = True) -> List[Dict]:
        """Advanced retrieval pipeline"""
        
        print(f"\n🔍 Query: {query}")
        
        expanded_queries = self.query_transformer.expand_query(query, num_variations=3)
        print(f"   Expanded to {len(expanded_queries)} variations")
        
        if use_hyde:
            hyde_answer = self.query_transformer.generate_hyde(query)
            expanded_queries.append(hyde_answer)
            print(f"   Generated HyDE answer")
        
        pre_filter = self.metadata_filter.build_pre_filter(query)
        if pre_filter:
            print(f"   Pre-filter: {pre_filter}")
        
        all_results = []
        seen_ids = set()
        
        for q in expanded_queries:
            embedding = self.embed_query(q)
            
            search_results = self.index.query(
                vector=embedding,
                top_k=25,
                filter=pre_filter if pre_filter else None,
                include_metadata=True
            )
            
            for match in search_results['matches']:
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    all_results.append({
                        'id': match['id'],
                        'score': match['score'],
                        'metadata': match['metadata']
                    })
        
        print(f"   Retrieved {len(all_results)} unique chunks")
        
        filtered_results = self.metadata_filter.post_filter_by_date(all_results)
        print(f"   After date filtering: {len(filtered_results)} chunks")
        
        reranked_results = self.reranker.rerank(
            query=query,
            documents=filtered_results,
            top_k=top_k
        )
        
        print(f"   Final top-{top_k} after reranking")
        
        return reranked_results


class RAGGenerator:
    """Generate answers using retrieved context"""
    
    def __init__(self, openai_client, model="gpt-4o-mini"):
        self.client = openai_client
        self.model = model
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved chunks into context"""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk['metadata']
            text = metadata['text']
            
            doc_id = metadata.get('document_id', 'Unknown')
            h1 = metadata.get('h1_header', '')
            h2 = metadata.get('h2_header', '')
            
            header_context = f" > {h1}" if h1 else ""
            if h2:
                header_context += f" > {h2}"
            
            context_parts.append(
                f"[Source {i}: {doc_id}{header_context}]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> Dict:
        """Generate answer with citations"""
        
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": "low"
            }
        
        context = self.format_context(retrieved_chunks)
        
        prompt = f"""You are a corporate policy assistant. Answer the user's question based ONLY on the provided policy documents.

RETRIEVED POLICY EXCERPTS:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question accurately using ONLY the information from the sources above
2. If the sources contain tables or numbers, include them in your answer
3. Cite sources using [Source N] notation
4. If the sources don't fully answer the question, say so clearly
5. Be concise but complete
6. Use professional corporate language

ANSWER:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        sources = []
        for chunk in retrieved_chunks:
            metadata = chunk['metadata']
            sources.append({
                "document_id": metadata.get('document_id', 'Unknown'),
                "document_title": metadata.get('document_title', 'Unknown'),
                "section": metadata.get('h1_header', '') + (
                    " > " + metadata.get('h2_header', '') if metadata.get('h2_header') else ''
                ),
                "rerank_score": chunk.get('rerank_score', 0)
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": "high" if len(retrieved_chunks) >= 3 else "medium"
        }


class ProjectAegisRAG:
    """Complete RAG system combining all components"""
    
    def __init__(self, openai_client, cohere_client, pinecone_index, embedding_model, llm_model):
        self.retriever = AdvancedRetriever(openai_client, cohere_client, pinecone_index, embedding_model)
        self.generator = RAGGenerator(openai_client, model=llm_model)
    
    def query(self, question: str, top_k: int = 5, use_hyde: bool = True) -> Dict:
        """End-to-end RAG query"""
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_hyde=use_hyde
        )
        
        result = self.generator.generate_answer(question, retrieved_chunks)
        
        return result
    
    def chat(self):
        """Interactive chat interface"""
        print("\n" + "="*60)
        print("🤖 Project Aegis RAG Chatbot")
        print("Type 'exit' to quit")
        print("="*60 + "\n")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = self.query(question)
            
            print(f"\n💡 Answer:\n{result['answer']}")
            print(f"\n📚 Sources ({len(result['sources'])})")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source['document_title']} - {source['section']} (Score: {source['rerank_score']:.3f})")
            print(f"\n🎯 Confidence: {result['confidence']}")
