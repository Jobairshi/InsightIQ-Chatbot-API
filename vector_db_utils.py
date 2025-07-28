# vector_db_utils.py
"""
Utility functions for managing Pinecone vector database operations
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import pinecone
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class VectorDBManager:
    """Advanced vector database management utilities"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index = pinecone.Index(index_name)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0),
                "namespaces": dict(stats.get('namespaces', {})),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def batch_upsert_vectors(self, vectors_data: List[Dict], batch_size: int = 100) -> Dict[str, Any]:
        """Batch upsert vectors with metadata"""
        try:
            total_vectors = len(vectors_data)
            upserted = 0
            
            for i in range(0, total_vectors, batch_size):
                batch = vectors_data[i:i + batch_size]
                
                # Prepare vectors for upsert
                vectors_to_upsert = []
                for vector_data in batch:
                    vectors_to_upsert.append({
                        "id": vector_data["id"],
                        "values": vector_data["values"],
                        "metadata": vector_data.get("metadata", {})
                    })
                
                # Upsert batch
                self.index.upsert(vectors=vectors_to_upsert)
                upserted += len(vectors_to_upsert)
                
                logger.info(f"Upserted batch {i//batch_size + 1}: {upserted}/{total_vectors} vectors")
            
            return {
                "status": "success",
                "total_upserted": upserted,
                "batches_processed": (total_vectors + batch_size - 1) // batch_size
            }
            
        except Exception as e:
            logger.error(f"Error in batch upsert: {e}")
            return {"status": "error", "error": str(e)}
    
    def semantic_search(self, query_vector: List[float], top_k: int = 10, 
                       filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Advanced semantic search with filtering"""
        try:
            search_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False
            }
            
            if filter_metadata:
                search_params["filter"] = filter_metadata
            
            results = self.index.query(**search_params)
            
            search_results = []
            for match in results.matches:
                search_results.append({
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": dict(match.metadata) if match.metadata else {}
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def delete_vectors_by_filter(self, filter_metadata: Dict) -> Dict[str, Any]:
        """Delete vectors matching metadata filter"""
        try:
            delete_result = self.index.delete(filter=filter_metadata)
            return {"status": "success", "deleted": True}
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return {"status": "error", "error": str(e)}
    
    def backup_metadata(self, output_file: str = None) -> Dict[str, Any]:
        """Backup vector metadata to file"""
        try:
            if not output_file:
                output_file = f"vector_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # This is a simplified backup - in production, you'd want to implement
            # proper pagination for large datasets
            stats = self.get_index_stats()
            
            backup_data = {
                "backup_timestamp": datetime.now().isoformat(),
                "index_name": self.index_name,
                "stats": stats
            }
            
            with open(output_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return {"status": "success", "backup_file": output_file}
            
        except Exception as e:
            logger.error(f"Error backing up metadata: {e}")
            return {"status": "error", "error": str(e)}

class RAGOptimizer:
    """Utilities for optimizing RAG performance"""
    
    def __init__(self):
        self.search_analytics = []
    
    def log_search_query(self, query: str, results_count: int, avg_score: float, 
                        response_time: float):
        """Log search query for analytics"""
        self.search_analytics.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results_count": results_count,
            "avg_score": avg_score,
            "response_time": response_time,
            "query_length": len(query.split())
        })
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search performance analytics"""
        if not self.search_analytics:
            return {"message": "No search data available"}
        
        df = pd.DataFrame(self.search_analytics)
        
        return {
            "total_searches": len(df),
            "avg_response_time": df['response_time'].mean(),
            "avg_results_count": df['results_count'].mean(),
            "avg_relevance_score": df['avg_score'].mean(),
            "query_length_stats": {
                "mean": df['query_length'].mean(),
                "median": df['query_length'].median(),
                "min": df['query_length'].min(),
                "max": df['query_length'].max()
            },
            "recent_queries": df.tail(5).to_dict('records')
        }
    
    def suggest_query_improvements(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """Suggest improvements for low-performing queries"""
        suggestions = []
        
        # Check if query is too short
        if len(query.split()) < 3:
            suggestions.append({
                "type": "query_expansion",
                "suggestion": "Try using more descriptive keywords",
                "example": f"Instead of '{query}', try adding context like 'how to {query}' or '{query} features'"
            })
        
        # Check if results have low scores
        if results and all(r.get('similarity_score', 0) < 0.7 for r in results):
            suggestions.append({
                "type": "low_relevance",
                "suggestion": "Try rephrasing your question or using different keywords",
                "example": "Use synonyms or more specific terms related to your topic"
            })
        
        # Check if no results
        if not results:
            suggestions.append({
                "type": "no_results",
                "suggestion": "Your query might be too specific or use unfamiliar terms",
                "example": "Try broader terms or check spelling"
            })
        
        return {
            "query": query,
            "suggestions": suggestions,
            "results_count": len(results)
        }

class EmbeddingQualityChecker:
    """Check the quality of embeddings and suggest improvements"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def check_embedding_distribution(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of embeddings"""
        try:
            # Calculate basic statistics
            mean_values = np.mean(embeddings, axis=0)
            std_values = np.std(embeddings, axis=0)
            
            # Check for potential issues
            issues = []
            
            # Check for dimensions with very low variance (potentially uninformative)
            low_variance_dims = np.where(std_values < 0.01)[0]
            if len(low_variance_dims) > embeddings.shape[1] * 0.1:  # More than 10% dimensions
                issues.append({
                    "type": "low_variance",
                    "description": f"{len(low_variance_dims)} dimensions have very low variance",
                    "impact": "These dimensions may not contribute much to similarity calculations"
                })
            
            # Check for potential clustering issues
            pairwise_similarity = np.corrcoef(embeddings)
            high_similarity_pairs = np.where(pairwise_similarity > 0.95)
            if len(high_similarity_pairs[0]) > len(embeddings) * 0.1:
                issues.append({
                    "type": "high_similarity",
                    "description": "Many embeddings are very similar to each other",
                    "impact": "This may indicate repetitive content or poor text preprocessing"
                })
            
            return {
                "total_embeddings": len(embeddings),
                "embedding_dimension": embeddings.shape[1],
                "mean_magnitude": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                "std_magnitude": float(np.std(np.linalg.norm(embeddings, axis=1))),
                "potential_issues": issues,
                "quality_score": max(0, 1.0 - len(issues) * 0.2)  # Simple quality score
            }
            
        except Exception as e:
            logger.error(f"Error checking embedding distribution: {e}")
            return {"error": str(e)}
    
    def suggest_preprocessing_improvements(self, text_data: List[str]) -> Dict[str, Any]:
        """Suggest text preprocessing improvements"""
        suggestions = []
        
        # Check average text length
        lengths = [len(text) for text in text_data]
        avg_length = np.mean(lengths)
        
        if avg_length < 50:
            suggestions.append({
                "type": "short_texts",
                "suggestion": "Texts are quite short, consider combining related entries",
                "details": f"Average length: {avg_length:.1f} characters"
            })
        elif avg_length > 2000:
            suggestions.append({
                "type": "long_texts",
                "suggestion": "Texts are quite long, consider chunking for better retrieval",
                "details": f"Average length: {avg_length:.1f} characters"
            })
        
        # Check for HTML content
        html_count = sum(1 for text in text_data if '<' in text and '>' in text)
        if html_count > len(text_data) * 0.1:
            suggestions.append({
                "type": "html_content",
                "suggestion": "Detected HTML content, ensure proper cleaning",
                "details": f"Found HTML in {html_count} out of {len(text_data)} texts"
            })
        
        # Check for very similar texts
        unique_texts = set(text_data)
        if len(unique_texts) < len(text_data) * 0.9:
            suggestions.append({
                "type": "duplicate_content",
                "suggestion": "Found duplicate or very similar texts",
                "details": f"Only {len(unique_texts)} unique texts out of {len(text_data)}"
            })
        
        return {
            "total_texts": len(text_data),
            "unique_texts": len(unique_texts),
            "avg_length": avg_length,
            "suggestions": suggestions
        }

def create_document_hash(content: str, metadata: Dict) -> str:
    """Create a unique hash for a document based on content and key metadata"""
    # Combine content with key metadata fields
    hash_content = content + str(metadata.get('id', '')) + str(metadata.get('title', ''))
    return hashlib.md5(hash_content.encode()).hexdigest()

def optimize_chunk_size(texts: List[str], target_chunks: int = None) -> Dict[str, Any]:
    """Suggest optimal chunk size based on text collection"""
    lengths = [len(text) for text in texts]
    total_chars = sum(lengths)
    avg_length = np.mean(lengths)
    
    if target_chunks:
        suggested_chunk_size = total_chars // target_chunks
    else:
        # Aim for chunks of reasonable size (500-1500 chars)
        if avg_length < 500:
            suggested_chunk_size = 500
        elif avg_length > 1500:
            suggested_chunk_size = 1000
        else:
            suggested_chunk_size = int(avg_length)
    
    return {
        "total_texts": len(texts),
        "total_characters": total_chars,
        "avg_text_length": avg_length,
        "suggested_chunk_size": suggested_chunk_size,
        "estimated_chunks": total_chars // suggested_chunk_size,
        "overlap_recommendation": max(50, suggested_chunk_size // 5)  # 20% overlap
    }