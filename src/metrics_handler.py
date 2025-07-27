import pandas as pd
import numpy as np
import json
import os

class EvaluationHandler:
    """
    Calculate Precision at k and NDCG at k metrics for the
    retrieved users ranking and based on the csv file that
    was used to build the knowledge graph.
    """
    def __init__(self, item_users_id_array, path ='data/watchedRel.csv'):
        self.item_users_id_array = item_users_id_array
        self.path = path

    def retrive_actual_users(self):
        """
        Retrieve the actual users who watched the movie (item). This
        is a public method to allow store the actual users in a variable
        for enhancing metrics calculation at different values of k.
        """
        actual_users = pd.read_csv(self.path, dtype={'userId': int, 'movieId': int})
        return actual_users[
            actual_users["movieId"].isin(self.item_users_id_array[0][0])]['userId'].to_numpy()

    def calculate_metrics(self, actual_users, k=100):
        """
        Calculate precision at k and ndcg at k for the retrieved users.
        """
        actual_users = actual_users
        precision_at_k = self._calculate_precision_at_k(actual_users, k)
        ndcg_at_k = self._calculate_ndcg_at_k(actual_users, k)
        return [precision_at_k, ndcg_at_k]

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
    Forat and save the report of the experiments with the hyperparameters,
    retrieval method and metrics. This class is used to save the report
    of the experiments in a JSON file and allow to retrieve the best configuration
    based on the highest value of a specific metric at a specific k.
    """
    def __init__(self, hyperparameters:dict, method:str, metrics:dict, exp_dir='experiments/'):
        self.hyperparameters = hyperparameters
        self.method = method
        self.metrics = metrics
        self.exp_dir = exp_dir

    def save_report(self, exp_id:str):
        experiment = {
            "experiment_id": exp_id,
            "hyperparams": self.hyperparameters,
            "retrieval_method": self.method,
            "metrics": self.metrics
        }

        os.makedirs(self.exp_dir, exist_ok=True)
        file_path = os.path.join(self.exp_dir, "item_cold_start_report.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(experiment)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    
    def get_best_config(self, metric: str, k=100):
        """
        Retrieve the best configuration based on the highest value of a specific metric at a specific k.
        Args:
            metric (str): The metric name to search for (e.g., "precision_at_k", "ndcg_at_k").
            k (int): The value of k to consider.
        Returns:
            dict: The experiment with the best metric value, or None if not found.
        """
        file_path = os.path.join(self.exp_dir, "item_cold_start_report.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        best_exp = None
        best_value = float('-inf')
        for exp in data:
            metrics = exp.get("metrics", {})
            # Support both list and dict metrics
            value = None
            if isinstance(metrics, dict):
                # If metrics are stored as dict: { "precision_at_k_10": 0.5, ... }
                key = f"{metric}_{k}" if not metric.endswith(f"_{k}") else metric
                value = metrics.get(key)
            elif isinstance(metrics, list):
                # If metrics are stored as list: [precision_at_k, ndcg_at_k, ...]
                # Assume order: [precision_at_k, ndcg_at_k]
                metric_map = {f"precision_at_k_{k}": 0, f"ndcg_at_k_{k}": 1}
                idx = metric_map.get(f"{metric}_{k}")
                if idx is not None and len(metrics) > idx:
                    value = metrics[idx]
            if value is not None and value > best_value:
                best_value = value
                best_exp = exp

        return best_exp 
