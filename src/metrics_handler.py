import pandas as pd
import numpy as np

class EvaluationHandler:
    """
    Calculate Precision at k and NDCG at k metrics for the
    retrieved users ranking and based on the csv file that
    was used to build the knowledge graph.
    """
    def __init__(self, item_users_id_array, path ='data/watchedRel.csv'):
        self.item_users_id_array = item_users_id_array
        self.path = path
        
    def calculate_metrics(self, k=100):
        """
        Calculate precision at k and ndcg at k for the retrieved users.
        """
        actual_users = self._retrive_actual_users()
        precision_at_k = self._calculate_precision_at_k(actual_users, k)
        ndcg_at_k = self._calculate_ndcg_at_k(actual_users, k)
        return [precision_at_k, ndcg_at_k]

    def _retrive_actual_users(self):
        """
        Retrieve the actual users who watched the movie (item).
        """
        actual_users = pd.read_csv(self.path, dtype={'userId': int, 'movieId': int})
        return actual_users[
            actual_users["movieId"].isin(self.item_users_id_array[0][0])]['userId'].to_numpy()

    def _calculate_precision_at_k(self, actual_users, k):
        """
        Calculate the precision at k for the retrieved users.
        """
        k = min(k, len(actual_users))
        retrieved_set = set(self.item_users_id_array[1][:k])
        actual_set = set(actual_users[:k])
        intersection = retrieved_set.intersection(actual_set)
        return round(len(intersection) / k if k > 0 else 0, 2)
    
    def _calculate_ndcg_at_k(self, actual_users, k):
        """
        Calculate the normalized discounted cumulative gain (NDCG) at k
        considering the retrieved ranking (DCG) and the perfect ranking (IDCG).
        """
        k = min(k, len(actual_users))
        retrieved_array = self.item_users_id_array[1][:k]
        actual_array = actual_users[:k]
        # Create binary relevance array for the retrieved users
        y_true = [1 if user in actual_array else 0 for user in retrieved_array]
        y_true_k = y_true[:k]

        # Calculate DCG for the retrieved users
        discounts_true = np.log2(np.arange(2, len(y_true_k) + 2))
        dcg = np.sum(y_true_k / discounts_true)

        # Calculate IDCG for the ideal retrieved ranking
        ideal_y = sorted(y_true_k, reverse=True)
        discounts_ideal = np.log2(np.arange(2, len(ideal_y) + 2))
        dcg_ideal = np.sum(ideal_y / discounts_ideal)
        return round(dcg / dcg_ideal if dcg_ideal > 0 else 0.0, 2)

class ReportHandler:
    """
    """