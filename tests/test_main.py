import sys
import os
import json
import types
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from main import main

# Import the main function from main.py using absolute import

@pytest.fixture
def mock_config_params(tmp_path, monkeypatch):
    # Create a temporary config_params.json file with a single combination of hyperparameters
    config = {
        "iterationWeights": [1.0],
        "paramA": 42,
        "paramB": "foo",
        "method": ["cosine"]
    }
    config_path = tmp_path / "config_params.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    monkeypatch.chdir(tmp_path)
    return config_path

@pytest.fixture
def mock_dependencies(monkeypatch):
    # Mock all imported classes used in main.py
    class DummyValidator:
        def __init__(self, **kwargs): pass

    class DummyCombinator:
        def __init__(self, validated): pass
        def generate_combinations(self): return [{
            "iterationWeights": [1.0],
            "paramA": 42,
            "paramB": "foo",
            "method": ["cosine"]
        }]

    class DummyNodeHandler:
        def __init__(self): pass
        def hold_and_remove_movies_sample(self):
            # Return dummy test_ids_names and test_ids_caracteristcs
            return ([{"movieId": 1}], {1: {"attr": "val"}})
        def recreate_movie_nodes(self, nodes): pass
        def recreate_movie_attribute_rels(self, attrs): pass
        def recreate_user_movie_rels(self, ids): pass

    class DummyNodeSubgraphHandler:
        def __init__(self, movie_id, len_hops): pass
        def create_node_subgraph_projection(self): return {}

    class DummyUserEmbeddingHandler:
        def __init__(self, params): pass
        def create_user_vectors_array(self): return (["u1", "u2"], [[0.1, 0.2], [0.2, 0.3]])

    class DummyItemEmbeddingHandler:
        def __init__(self, movie_id, subgraph, params): pass
        def create_item_vectors_array(self): return ([1], [[0.5, 0.5]])

    class DummyVectorRetriever:
        def __init__(self, item_array, user_array, method, length): pass
        def retrieve_users(self): return (1, ["u1", "u2"])

    class DummyEvaluationHandler:
        def __init__(self, rec_users): pass
        def retrive_actual_users(self): return ["u1", "u2"]
        def calculate_metrics(self, real_users): return (1.0, 0.9)

    class DummyReportHandler:
        def __init__(self, timestamp): pass
        def save_report(self, **kwargs): pass

    monkeypatch.setattr("main.HyperparamValidator", DummyValidator)
    monkeypatch.setattr("main.HyperparamCombinator", DummyCombinator)
    monkeypatch.setattr("main.NodeHandler", DummyNodeHandler)
    monkeypatch.setattr("main.NodeSubgraphHandler", DummyNodeSubgraphHandler)
    monkeypatch.setattr("main.UserEmbeddingHandler", DummyUserEmbeddingHandler)
    monkeypatch.setattr("main.ItemEmbeddingHandler", DummyItemEmbeddingHandler)
    monkeypatch.setattr("main.VectorRetriever", DummyVectorRetriever)
    monkeypatch.setattr("main.EvaluationHandler", DummyEvaluationHandler)
    monkeypatch.setattr("main.ReportHandler", DummyReportHandler)

def test_main_runs_successfully(mock_config_params, mock_dependencies, monkeypatch):
    # Patch sys.argv so main() doesn't get confused
    monkeypatch.setattr(sys, "argv", ["main.py"])
    # Should not raise any exceptions
    assert main() is None