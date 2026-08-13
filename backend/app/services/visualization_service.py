"""
File: backend/app/services/visualization_service.py
Description: Service for rendering UMAP embeddings visualizations and saving debug plots.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import umap

logger = logging.getLogger(__name__)


class VisualizationService:
    def __init__(self, debug_dir: str = "storage/debug"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def generate_umap_plot(
        self, 
        query_embedding: List[float], 
        retrieved_embeddings: List[List[float]], 
        background_embeddings: List[List[float]],
        query_text: str = "Query",
        filename: Optional[str] = None  # <--- NEW PARAMETER
    ) -> str:
        
        # 1. Logging & Diagnostics
        logger.info(f"[UMAP] Background vectors count: {len(background_embeddings)}")
        logger.info(f"[UMAP] Retrieved vectors count: {len(retrieved_embeddings)}")

        if not background_embeddings or len(background_embeddings) < 2:
            raise ValueError(
                f"Cannot build UMAP projection. ChromaDB returned {len(background_embeddings)} background vectors. "
                f"Ensure you have uploaded PDFs to ChromaDB before running debug visualization."
            )

        # 2. Convert to Numpy Arrays
        dataset_arr = np.array(background_embeddings)
        query_arr = np.array([query_embedding])

        # 3. Fit UMAP on the overall vector database
        n_neighbors = min(15, len(background_embeddings) - 1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            metric='cosine',
            random_state=42,
            transform_seed=42
        )
        projected_dataset = reducer.fit_transform(dataset_arr)

        # 4. Project Query and Retrieved Embeddings into the learned space
        projected_query = reducer.transform(query_arr)

        projected_retrieved = None
        if retrieved_embeddings and len(retrieved_embeddings) > 0:
            retrieved_arr = np.array(retrieved_embeddings)
            projected_retrieved = reducer.transform(retrieved_arr)

        # 5. Render Plot
        plt.figure(figsize=(10, 8))

        # Layer 1: Background Dataset (Gray Dots)
        plt.scatter(
            projected_dataset[:, 0], projected_dataset[:, 1], 
            c='gray', s=20, alpha=0.5, label='Database Chunks'
        )

        # Layer 2: Top-K Retrieved Chunks (Green Open Circles)
        if projected_retrieved is not None and len(projected_retrieved) > 0:
            plt.scatter(
                projected_retrieved[:, 0], projected_retrieved[:, 1], 
                s=120, facecolors='none', edgecolors='g', linewidths=2, label='Retrieved Chunks (Top K)'
            )

        # Layer 3: Query Vector (Red 'X')
        plt.scatter(
            projected_query[:, 0], projected_query[:, 1], 
            c='r', marker='X', s=200, label='Query'
        )

        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f"Vector Space Evaluation\nQuery: '{query_text[:50]}...'")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.3)

        # 6. Save Plot with pre-determined filename (or fallback timestamp)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"umap_query_{timestamp}.png"
            
        filepath = self.debug_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)