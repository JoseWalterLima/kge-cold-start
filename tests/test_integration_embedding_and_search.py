import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler
from vector_search_handler import VectorRetriever

@patch("embedding_handler.get_gds_connection")
def test_embedding_and_vector_search_integration(mock_get_gds):
    # Mock GDS for embeddings
    gds = MagicMock()
    gds.graph.project.return_value = ("graph_proj", "metadata")
    gds.fastRP.stream.return_value = pd.DataFrame({
        "nodeId": [1, 2, 3],
        "embedding": [np.array([0.1, 0.2]), np.array([0.9, 0.8]), np.array([0.2, 0.1])]
    })
    gds.run_cypher.return_value = pd.DataFrame({
        "nodeId": [1, 2, 3],
        "userId": [101, 102, 103]
    })
    mock_get_gds.return_value = gds

    # User embeddings
    user_handler = UserEmbeddingHandler(params={"embeddingDimension": 2})
    user_handler.node_projection = {"User": {"label": "User"}}
    user_handler.relationship_projection = {"REL": {"type": "REL", "orientation": "UNDIRECTED"}}
    user_vectors_array = user_handler.create_user_vectors_array()

    # Item embedding (simulate for item with nodeId=2)
    item_handler = ItemEmbeddingHandler(
        subgraph_projection="subgraph_proj",
        target_node_id=2,
        params={"embeddingDimension": 2}
    )
    item_handler.gds = gds  # Use the same mock
    item_handler.gds.fastRP.stream.return_value = pd.DataFrame({
        "nodeId": [1, 2, 3],
        "embedding": [np.array([0.1, 0.2]), np.array([0.8, 0.7]), np.array([0.2, 0.1])]
    })
    item_handler.gds.run_cypher.return_value = pd.DataFrame({"nodeId": [2]})
    item_id = 2
    item_vector_array = item_handler.create_item_vector_array()

    # Vector search
    retriever = VectorRetriever(item_vector_array, user_vectors_array, method='cosine', length=2)
    returned_item_user_ids = retriever.retrieve_users()

    assert returned_item_user_ids[0] == 2
    assert returned_item_user_ids[1][0] == 102
    assert len(returned_item_user_ids[1]) == 2
    assert all(uid for uid in returned_item_user_ids[1])