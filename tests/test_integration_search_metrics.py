import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
import pandas as pd
import tempfile
from vector_search_handler import VectorRetriever
from metrics_handler import EvaluationHandler

def test_vector_search_and_metrics_integration():
    # Simulate user ids and embeddings
    user_ids = np.array([101, 102, 103, 104, 105])
    user_vectors = np.array([
        [0.1, 0.2],
        [0.9, 0.8],
        [0.2, 0.1],
        [0.5, 0.5],
        [0.3, 0.7]
    ])
    users_array = (user_ids, user_vectors)

    # Simulate item id and embedding
    item_id = np.array([201])
    item_vector = np.array([[0.8, 0.7]])
    item_array = (item_id, item_vector)

    # Retrieve top 3 users using cosine similarity
    retriever = VectorRetriever(item_array, users_array, method='cosine', length=3)
    returned_item_id, ordered_user_ids = retriever.retrieve_users()

    # Create a temporary CSV file to simulate watchedRel.csv
    data = pd.DataFrame({
        "userId": [101, 102, 106, 107],
        "movieId": [201, 201, 201, 202]
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        data.to_csv(tmp.name, index=False)
        test_path = tmp.name

    # Prepare item_users_id_array for metrics handler
    item_users_id_array = [
        [[201]],  # item id
        list(ordered_user_ids)  # retrieved user ids
    ]

    handler = EvaluationHandler(item_users_id_array, path=test_path)
    actual_users = handler.retrive_actual_users()
    precision, ndcg = handler.calculate_metrics(actual_users, k=3)

    # There are one relevant users (101) in the top 3 retrieved (102, 104, 105)
    assert 0.33 <= precision <= 0.34
    assert 0.9 <= ndcg <= 1.0

    # Clean up temp file
    os.remove(test_path)
