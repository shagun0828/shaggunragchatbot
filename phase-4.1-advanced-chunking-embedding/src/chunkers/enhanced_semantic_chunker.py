"""
Enhanced Semantic Chunker
Advanced semantic chunking with multiple similarity metrics and adaptive thresholds
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

from models.chunk import Chunk


class EnhancedSemanticChunker:
    """Enhanced semantic chunker with multiple similarity metrics and adaptive thresholds"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Chunking parameters
        self.base_similarity_threshold = 0.7
        self.max_chunk_size = 1000
        self.min_chunk_size = 100
        self.max_chunk_overlap = 0.3
        
        # Adaptive parameters
        self.adaptive_threshold = True
        self.diversity_penalty = 0.1
        self.length_penalty = 0.05
        
        # TF-IDF for lexical similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def chunk_semantic_enhanced(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Enhanced semantic chunking with multiple similarity metrics"""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [self._create_chunk(text, metadata)]
        
        # Generate embeddings
        embeddings = self.model.encode(sentences)
        
        # Calculate multiple similarity matrices
        semantic_sim_matrix = self._calculate_semantic_similarity(embeddings)
        lexical_sim_matrix = self._calculate_lexical_similarity(sentences)
        positional_sim_matrix = self._calculate_positional_similarity(len(sentences))
        
        # Combine similarity matrices
        combined_similarity = self._combine_similarities(
            semantic_sim_matrix, lexical_sim_matrix, positional_sim_matrix
        )
        
        # Apply adaptive thresholding
        if self.adaptive_threshold:
            similarity_threshold = self._calculate_adaptive_threshold(combined_similarity)
        else:
            similarity_threshold = self.base_similarity_threshold
        
        # Perform chunking using TextRank algorithm
        chunks = self._textrank_chunking(
            sentences, combined_similarity, similarity_threshold, metadata
        )
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting with financial terminology awareness"""
        # Financial sentence patterns
        financial_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence boundaries
            r'(?<=[.!?])\s+(?=\d)',     # Sentences starting with numbers
            r'(?<=[%])\s+(?=[A-Z])',    # After percentages
            r'(?<=[Rsâ¹â¹])\s+(?=[A-Z])', # After currency symbols
        ]
        
        sentences = [text]
        for pattern in financial_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_semantic_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate semantic similarity matrix"""
        return cosine_similarity(embeddings)
    
    def _calculate_lexical_similarity(self, sentences: List[str]) -> np.ndarray:
        """Calculate lexical similarity using TF-IDF"""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            return cosine_similarity(tfidf_matrix)
        except Exception:
            # Fallback to zero matrix if TF-IDF fails
            return np.zeros((len(sentences), len(sentences)))
    
    def _calculate_positional_similarity(self, num_sentences: int) -> np.ndarray:
        """Calculate positional similarity (sentences closer together have higher similarity)"""
        pos_sim = np.zeros((num_sentences, num_sentences))
        for i in range(num_sentences):
            for j in range(num_sentences):
                if i != j:
                    distance = abs(i - j)
                    pos_sim[i][j] = np.exp(-distance / 5.0)  # Exponential decay
        return pos_sim
    
    def _combine_similarities(self, semantic: np.ndarray, lexical: np.ndarray, 
                           positional: np.ndarray) -> np.ndarray:
        """Combine multiple similarity matrices with weights"""
        # Dynamic weighting based on matrix quality
        semantic_weight = 0.6
        lexical_weight = 0.3
        positional_weight = 0.1
        
        combined = (semantic_weight * semantic + 
                   lexical_weight * lexical + 
                   positional_weight * positional)
        
        return combined
    
    def _calculate_adaptive_threshold(self, similarity_matrix: np.ndarray) -> float:
        """Calculate adaptive similarity threshold based on data distribution"""
        # Get upper triangle (excluding diagonal)
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        if len(similarities) == 0:
            return self.base_similarity_threshold
        
        # Calculate statistics
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Adaptive threshold: mean + 0.5 * std
        adaptive_threshold = mean_sim + 0.5 * std_sim
        
        # Clamp to reasonable range
        adaptive_threshold = np.clip(adaptive_threshold, 0.3, 0.9)
        
        return adaptive_threshold
    
    def _textrank_chunking(self, sentences: List[str], similarity_matrix: np.ndarray, 
                          threshold: float, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Use TextRank algorithm for chunking"""
        # Build graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply threshold (remove weak edges)
        edges_to_remove = []
        for i, j, data in graph.edges(data=True):
            if data['weight'] < threshold:
                edges_to_remove.append((i, j))
        
        graph.remove_edges_from(edges_to_remove)
        
        # Find connected components (potential chunks)
        chunks = []
        for component_nodes in nx.connected_components(graph):
            if len(component_nodes) == 0:
                continue
            
            # Sort nodes to maintain order
            sorted_nodes = sorted(component_nodes)
            
            # Extract sentences for this chunk
            chunk_sentences = [sentences[i] for i in sorted_nodes]
            chunk_text = " ".join(chunk_sentences)
            
            # Check chunk size constraints
            if len(chunk_text) < self.min_chunk_size:
                # Try to merge with adjacent chunks
                chunk_text = self._expand_small_chunk(chunk_text, sentences, sorted_nodes)
            
            if len(chunk_text) <= self.max_chunk_size and len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = {
                    'chunk_type': 'enhanced_semantic',
                    'similarity_threshold': threshold,
                    'component_size': len(component_nodes),
                    'avg_similarity': np.mean([similarity_matrix[i][j] for i in component_nodes for j in component_nodes if i != j]) if len(component_nodes) > 1 else 0,
                    **(metadata or {})
                }
                chunks.append(self._create_chunk(chunk_text, chunk_metadata))
        
        return chunks
    
    def _expand_small_chunk(self, chunk_text: str, all_sentences: List[str], 
                           current_nodes: List[int]) -> str:
        """Expand small chunks by adding adjacent sentences"""
        # Find adjacent sentences not in current chunk
        min_node = min(current_nodes)
        max_node = max(current_nodes)
        
        expanded_sentences = [all_sentences[i] for i in current_nodes]
        
        # Add sentences before
        for i in range(min_node - 1, -1, -1):
            if len(" ".join(expanded_sentences + [all_sentences[i]])) <= self.max_chunk_size:
                expanded_sentences.insert(0, all_sentences[i])
            else:
                break
        
        # Add sentences after
        for i in range(max_node + 1, len(all_sentences)):
            if len(" ".join(expanded_sentences + [all_sentences[i]])) <= self.max_chunk_size:
                expanded_sentences.append(all_sentences[i])
            else:
                break
        
        return " ".join(expanded_sentences)
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process chunks to improve quality"""
        processed_chunks = []
        
        for chunk in chunks:
            # Remove redundant whitespace
            text = re.sub(r'\s+', ' ', chunk.text.strip())
            
            # Check for meaningful content
            if self._has_meaningful_content(text):
                chunk.text = text
                chunk.metadata['post_processed'] = True
                chunk.metadata['text_length'] = len(text)
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if chunk contains meaningful content"""
        # Check minimum content requirements
        if len(text) < self.min_chunk_size:
            return False
        
        # Check for meaningful patterns
        meaningful_patterns = [
            r'\b\d+\.?\d*%?\b',  # Numbers/percentages
            r'\b[A-Z][a-z]+\b',  # Proper nouns
            r'\b\w{4,}\b',       # Words longer than 3 characters
        ]
        
        pattern_matches = sum(1 for pattern in meaningful_patterns if re.search(pattern, text))
        return pattern_matches >= 2
    
    def _create_chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Chunk:
        """Create a chunk with enhanced metadata"""
        chunk_metadata = {
            'chunk_type': 'enhanced_semantic',
            'model_dimension': self.dimension,
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            **(metadata or {})
        }
        return Chunk(text=text, metadata=chunk_metadata)
