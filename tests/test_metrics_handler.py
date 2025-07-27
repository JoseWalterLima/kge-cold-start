import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
import pandas as pd
import tempfile
from metrics_handler import EvaluationHandler

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
    precision, ndcg = handler.calculate_metrics(k=3)

    # For k=3, actual users for movie 10 are [1,2,3]
    # retrieved: [2,6,1] -> intersection: [1,2] => precision = 2/3 ≈ 0.67
    # retrieved: [1,0,1] -> NDCG ≈ 0.91
    assert 0.66 <= precision <= 0.67  # rounding may vary
    assert 0.91 <= ndcg <= 0.92

    # Clean up temporary files
    os.remove(test_path)
