import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from node_handler import NodeHandler, NodeSubgraphHandler

def mock_gds():
    gds = MagicMock()
    gds.run_cypher.return_value = pd.DataFrame({
        "id": ["1", "2", "3"],
        "movieId": ["1", "2", "3"],
        "movieTitle": ["A", "B", "C"],
        "relType": ["HAS_GENRE", "HAS_GENRE", "HAS_GENRE"],
        "nodeLabel": ["Genre", "Genre", "Genre"],
        "genreDesc": ["Action", "Comedy", "Drama"],
        "releaseDate": ["2000", "2001", "2002"],
        "total": [3, 3, 3]
    })
    gds.graph.project.cypher.return_value = ("subgraph_proj", "metadata")
    return gds

@patch("node_handler.get_gds_connection")
def test_sampling_movie_nodes(mock_get_gds):
    gds = MagicMock()
    gds.run_cypher.side_effect = [
        pd.DataFrame({"total": [100]}),  # For count
        pd.DataFrame({"id": ["1", "2", "3"]})  # For sampling
    ]
    mock_get_gds.return_value = gds
    handler = NodeHandler()
    result = handler.sampling_movie_nodes(sample_ratio=0.03)
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)

@patch("node_handler.get_gds_connection")
def test_extract_movie_nodes_relations(mock_get_gds):
    mock_get_gds.return_value = MagicMock(
        run_cypher=MagicMock(return_value=[
            {
                "movieId": "1",
                "movieTitle": "A",
                "relType": "HAS_GENRE",
                "nodeLabel": "Genre",
                "genreDesc": "Action",
                "releaseDate": "2000"
            }
        ])
    )
    handler = NodeHandler()
    movies_dict, groups = handler.extract_movie_nodes_relations(["1"])
    assert isinstance(movies_dict, list)
    assert hasattr(groups, "groups")

@patch("node_handler.get_gds_connection")
def test_delete_nodes_and_rels(mock_get_gds):
    gds = MagicMock()
    mock_get_gds.return_value = gds
    handler = NodeHandler()
    handler.delete_nodes_and_rels(["1", "2"])
    gds.run_cypher.assert_called()

@patch("node_handler.get_gds_connection")
def test_recreate_movie_nodes(mock_get_gds):
    gds = MagicMock()
    mock_get_gds.return_value = gds
    handler = NodeHandler()
    handler.recreate_movie_nodes([{"movieId": "1", "movieTitle": "A"}])
    gds.run_cypher.assert_called()

@patch("node_handler.get_gds_connection")
def test_recreate_movie_attribute_rels(mock_get_gds):
    gds = MagicMock()
    mock_get_gds.return_value = gds
    handler = NodeHandler()
    group = pd.DataFrame({
        "movieId": ["1"],
        "attributeValue": ["Action"]
    })
    groups = [(("Genre", "genreDesc", "HAS_GENRE"), group)]
    handler.recreate_movie_attribute_rels(groups)
    gds.run_cypher.assert_called()

@patch("node_handler.get_gds_connection")
def test_recreate_user_movie_rels(mock_get_gds, tmp_path):
    gds = MagicMock()
    mock_get_gds.return_value = gds
    handler = NodeHandler()
    # Create a temporary CSV file
    csv_path = tmp_path / "watchedRel.csv"
    df = pd.DataFrame({"userId": ["1", "2"], "movieId": ["1", "2"]})
    df.to_csv(csv_path, index=False)
    handler.recreate_user_movie_rels(["1"], csv_path=str(csv_path))
    gds.run_cypher.assert_called()

@patch("node_handler.get_gds_connection")
def test_create_node_subgraph_projection(mock_get_gds):
    gds = MagicMock()
    gds.run_cypher.return_value = pd.DataFrame({"id": [1, 2, 3]})
    gds.graph.project.cypher.return_value = ("subgraph_proj", "metadata")
    mock_get_gds.return_value = gds
    handler = NodeSubgraphHandler(movie_id="1", hops=2)
    proj = handler.create_node_subgraph_projection()
    assert proj