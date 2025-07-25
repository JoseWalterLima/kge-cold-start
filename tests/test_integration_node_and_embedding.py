import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from node_handler import NodeHandler, NodeSubgraphHandler
from embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler

@patch("node_handler.get_gds_connection")
@patch("embedding_handler.get_gds_connection")
def test_integration_user_embedding_with_node_sampling(mock_gds_emb, mock_gds_node):
    # Mock GDS for NodeHandler
    gds_node = MagicMock()
    gds_node.run_cypher.side_effect = [
        pd.DataFrame({"total": [100]}),  # For count
        pd.DataFrame({"id": ["1", "2", "3"]})  # For sampling
    ]
    mock_gds_node.return_value = gds_node

    # Mock GDS for UserEmbeddingHandler
    gds_emb = MagicMock()
    gds_emb.graph.project.return_value = ("graph_proj", "metadata")
    gds_emb.fastRP.stream.return_value = pd.DataFrame({
        "nodeId": [1, 2],
        "embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    })
    gds_emb.run_cypher.return_value = pd.DataFrame({
        "nodeId": [1, 2],
        "userId": [10, 20]
    })
    mock_gds_emb.return_value = gds_emb

    # Use NodeHandler to sample movies
    node_handler = NodeHandler()
    sampled_movie_ids = node_handler.sampling_movie_nodes(sample_ratio=0.02)
    assert isinstance(sampled_movie_ids, list)
    assert len(sampled_movie_ids) > 0

    # Use UserEmbeddingHandler to get user embeddings
    user_emb_handler = UserEmbeddingHandler(
        node_projection={"User": {"label": "User"}},
        relationship_projection={"REL": {"type": "REL", "orientation": "UNDIRECTED"}},
        params={"embeddingDimension": 2}
    )
    user_ids, embeddings = user_emb_handler.create_user_vectors()
    assert isinstance(user_ids, list)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[1] == 2

@patch("node_handler.get_gds_connection")
@patch("embedding_handler.get_gds_connection")
def test_integration_item_embedding_with_subgraph(mock_gds_emb, mock_gds_node):
    # Mock GDS for NodeSubgraphHandler
    gds_node = MagicMock()
    gds_node.run_cypher.return_value = pd.DataFrame({"id": [1, 2, 3]})
    gds_node.graph.project.cypher.return_value = ("subgraph_proj", "metadata")
    mock_gds_node.return_value = gds_node

    # Mock GDS for ItemEmbeddingHandler
    gds_emb = MagicMock()
    gds_emb.fastRP.stream.return_value = pd.DataFrame({
        "nodeId": [1, 2, 3],
        "embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])]
    })
    mock_gds_emb.return_value = gds_emb

    # Use NodeSubgraphHandler to create a subgraph projection
    subgraph_handler = NodeSubgraphHandler(movie_id="1", hops=2)
    projection = subgraph_handler.create_node_subgraph_projection()
    assert projection == "subgraph_proj"

    # Use ItemEmbeddingHandler to get item embedding for a node in the subgraph
    item_emb_handler = ItemEmbeddingHandler(
        subgraph_projection=projection,
        target_node_id=2,
        params={"embeddingDimension": 2}
    )
    embedding = item_emb_handler.create_item_vector()
    assert np.allclose(embedding, np.array([0.3, 0.4]))