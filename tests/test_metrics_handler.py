import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
import pandas as pd
import tempfile
from metrics_handler import EvaluationHandler, ReportHandler

def test_evaluationhandler_metrics():
    # Create a temporary CSV file to simulate watchedRel.csv
    data = pd.DataFrame({
        "userId": [1, 2, 3, 4, 5],
        "movieId": [10, 10, 10, 20, 20]
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        data.to_csv(tmp.name, index=False)
        test_path = tmp.name

    # Simulate item_users_id_array for movieId 10
    # Format: [ [ [movieId] ], [retrieved_user_ids] ]
    item_users_id_array = [
        [[10]],  # item id
        [2, 6, 1, 7, 3]  # retrieved user ids (top 5)
    ]

    handler = EvaluationHandler(item_users_id_array, path=test_path)
    actual_users = handler.retrive_actual_users()
    assert np.array_equal(actual_users, np.array([1, 2, 3]))
    precision, ndcg = handler.calculate_metrics(actual_users, k=3)

    # For k=3, actual users for movie 10 are [1,2,3]
    # retrieved: [2,6,1] -> intersection: [1,2] => precision = 2/3 ≈ 0.67
    # retrieved: [1,0,1] -> NDCG ≈ 0.91
    assert 0.66 <= precision <= 0.67  # rounding may vary
    assert 0.91 <= ndcg <= 0.92

    # Clean up temporary files
    os.remove(test_path)

def test_reporthandler_save_and_get_best_config(tmp_path):
    # Prepare test data
    exp_dir = tmp_path / "experiments"
    os.makedirs(exp_dir, exist_ok=True)
    metrics1 = {"precision_at_k_10": 0.5, "ndcg_at_k_10": 0.7}
    metrics2 = {"precision_at_k_10": 0.8, "ndcg_at_k_10": 0.9}
    rh1 = ReportHandler({"lr": 0.01}, "methodA", metrics1, exp_dir=str(exp_dir))
    rh2 = ReportHandler({"lr": 0.02}, "methodB", metrics2, exp_dir=str(exp_dir))

    # Save two experiment reports
    rh1.save_report("exp1")
    rh2.save_report("exp2")

    # Test get_best_config for precision
    best = rh1.get_best_config("precision_at_k", k=10)
    assert best["experiment_id"] == "exp2"
    assert best["metrics"]["precision_at_k_10"] == 0.8

    # Test get_best_config for ndcg
    best = rh1.get_best_config("ndcg_at_k", k=10)
    assert best["experiment_id"] == "exp2"
    assert best["metrics"]["ndcg_at_k_10"] == 0.9

def test_reporthandler_save_and_get_best_config_with_list_metrics(tmp_path):
    # Prepare test data
    exp_dir = tmp_path / "experiments"
    os.makedirs(exp_dir, exist_ok=True)
    metrics1 = [0.4, 0.6]  # [precision_at_k_10, ndcg_at_k_10]
    metrics2 = [0.9, 0.8]
    rh1 = ReportHandler({"param": 1}, "methodX", metrics1, exp_dir=str(exp_dir))
    rh2 = ReportHandler({"param": 2}, "methodY", metrics2, exp_dir=str(exp_dir))

    # Save two experiment reports
    rh1.save_report("expA")
    rh2.save_report("expB")

    # Test get_best_config for precision
    best = rh1.get_best_config("precision_at_k", k=10)
    assert best["experiment_id"] == "expB"
    assert best["metrics"][0] == 0.9

    # Test get_best_config for ndcg
    best = rh1.get_best_config("ndcg_at_k", k=10)
    assert best["experiment_id"] == "expB"
    assert best["metrics"][1] == 0.8
