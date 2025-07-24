import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler

def make_mock_gds():
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
def test_user_embedding_handler(mock_get_gds):
    mock_get_gds.return_value = make_mock_gds()
    handler = UserEmbeddingHandler(
        node_projection={"User": {"label": "User"}},
        relationship_projection={"REL": {"type": "REL", "orientation": "UNDIRECTED"}},
        params={"embeddingDimension": 2}
    )
    user_ids, embeddings = handler.create_user_vectors()
    assert user_ids == [10, 20]
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 2)

@patch("embedding_handler.get_gds_connection")
def test_item_embedding_handler(mock_get_gds):
    mock_get_gds.return_value = make_mock_gds()
    handler = ItemEmbeddingHandler(
        subgraph_projection="subgraph_proj",
        target_node_id=2,
        params={"embeddingDimension": 2}
    )
    # Simulate DataFrame as returned by fastRP.stream
    handler.gds.fastRP.stream.return_value = pd.DataFrame({
        "nodeId": [1, 2],
        "embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    })
    embedding = handler.create_item_vector()
    assert np.allclose(embedding, np.array([0.3, 0.4]))