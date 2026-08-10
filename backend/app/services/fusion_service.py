"""
File: backend/app/services/fusion_service.py
Description: This service takes multiple lists of search results (Candidate Chunks) 
and mathematically merges them using Reciprocal Rank Fusion (RRF). 
Chunks that appear highly ranked across multiple different queries get the best scores.
"""

from typing import List, Dict, Any
import logging

# Assuming you have a Pydantic model for chunks. Adjust the import based on your setup.
from app.models.domain import RetrievedChunk 

logger = logging.getLogger(__name__)

class FusionService:
    def __init__(self, rrf_k: int = 60):
        """
        Initialize the Fusion Service.
        
        Args:
            rrf_k: The smoothing constant for the RRF mathematical formula. 
                   60 is the industry standard default.
        """
        self.rrf_k = rrf_k

    def fuse_results(self, multiple_result_lists: List[List[RetrievedChunk]], top_k: int = 5) -> List[RetrievedChunk]:
        """
        Applies Reciprocal Rank Fusion to merge and rank duplicate chunks.
        
        Args:
            multiple_result_lists: A list where each item is a list of chunks returned from ChromaDB.
            top_k: The final number of chunks we want to return to the LLM.
            
        Returns:
            A single, flat, ranked list of the best chunks.
        """
        
        # A dictionary to keep track of the accumulated RRF score for each unique chunk.
        # Key: chunk_id (string), Value: the accumulated float score.
        rrf_scores: Dict[str, float] = {}
        
        # A dictionary to store the actual chunk objects so we can retrieve them later.
        # Key: chunk_id (string), Value: RetrievedChunk object.
        chunk_map: Dict[str, RetrievedChunk] = {}

        # Loop through each query's result list
        for result_list in multiple_result_lists:
            
            # Loop through the chunks inside that specific result list
            # enumerate(..., start=1) gives us the position (rank 1, rank 2, etc.)
            for rank, chunk in enumerate(result_list, start=1):
                
                # Save the chunk to our map if we haven't seen it yet
                if chunk.chunk_id not in chunk_map:
                    chunk_map[chunk.chunk_id] = chunk
                    rrf_scores[chunk.chunk_id] = 0.0
                
                # Calculate the RRF score using the formula: 1 / (K + rank)
                score_addition = 1.0 / (self.rrf_k + rank)
                
                # Add the score to the chunk's running total
                rrf_scores[chunk.chunk_id] += score_addition

        # Now that every chunk is scored, we sort them from highest score to lowest score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # Rebuild the final list using the sorted IDs
        final_ranked_chunks = []
        for cid in sorted_chunk_ids:
            chunk = chunk_map[cid]
            # Optional: Overwrite the ChromaDB distance score with our new RRF score
            chunk.score = rrf_scores[cid] 
            final_ranked_chunks.append(chunk)

        # Slice the array to return only the absolute top N chunks
        final_top_k = final_ranked_chunks[:top_k]
        
        logger.info(f"RRF applied. Merged {len(chunk_map)} total unique chunks down to top {len(final_top_k)}.")
        return final_top_k