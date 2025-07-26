import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from vector_search_handler import VectorRetriever

def test_vector_retriever_cosine():
    # Simulate user ids and embeddings
    user_ids = np.array([101, 102, 103])
    user_vectors = np.array([
        [0.1, 0.2],
        [0.9, 0.8],
        [0.2, 0.1]
    ])
    users_array = (user_ids, user_vectors)

    # Simulate item id and embedding
    item_id = np.array([201])
    item_vector = np.array([[0.8, 0.7]])
    item_array = (item_id, item_vector)

    retriever = VectorRetriever(item_array, users_array, method='cosine', length=2)
    returned_item_id, ordered_user_ids = retriever.retrieve_users()

    assert returned_item_id == 201
    assert len(ordered_user_ids) == 2
    assert all(uid in user_ids for uid in ordered_user_ids)

def test_vector_retriever_euclidean():
    user_ids = np.array([201, 202, 203])
    user_vectors = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5]
    ])
    users_array = (user_ids, user_vectors)

    item_id = np.array([301])
    item_vector = np.array([[0.6, 0.6]])
    item_array = (item_id, item_vector)

    retriever = VectorRetriever(item_array, users_array, method='euclidean', length=2)
    returned_item_id, ordered_user_ids = retriever.retrieve_users()

    assert returned_item_id == 301
    assert len(ordered_user_ids) == 2
    assert all(uid in user_ids for uid in ordered_user_ids)
