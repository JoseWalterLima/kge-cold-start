import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler

def mock_gds():
    gds = MagicMock()
    gds.graph.project.return_value = ("graph_proj", "metadata")
    gds.fastRP.stream.return_value = pd.DataFrame({
        "nodeId": [1, 2],
        "embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    })
    gds.run_cypher.return_value = pd.DataFrame({
        "nodeId": [1, 2],
        "userId": [10, 20]
    })
    return gds

@patch("embedding_handler.get_gds_connection")
def test_user_embedding_handler(create_gds):
    create_gds.return_value = mock_gds()
    handler = UserEmbeddingHandler(params={"embeddingDimension": 2})
    handler.node_projection = {"User": {"label": "User"}}
    handler.relationship_projection = {"REL": {"type": "REL", "orientation": "UNDIRECTED"}}
    user_ids, embeddings = handler.create_user_vectors_array()
    assert np.array_equal(user_ids, [10, 20])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 2)

@patch("embedding_handler.get_gds_connection")
def test_item_embedding_handler(create_gds):
    gds = mock_gds()
    gds.run_cypher.return_value = pd.DataFrame({"nodeId": [2]})
    create_gds.return_value = gds
    handler = ItemEmbeddingHandler(
        subgraph_projection="subgraph_proj",
        target_node_id=2,
        params={"embeddingDimension": 2}
    )
    handler.gds.fastRP.stream.return_value = pd.DataFrame({
        "nodeId": [1, 2],
        "embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    })
    item_id, embedding = handler.create_item_vector_array()
    assert item_id == 2
    assert np.allclose(embedding, np.array([0.3, 0.4]))
