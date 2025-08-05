import sys
import json
import time
from params_parser import HyperparamValidator, HyperparamCombinator
from node_handler import NodeHandler, NodeSubgraphHandler
from embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler
from vector_search_handler import VectorRetriever
from metrics_handler import EvaluationHandler, ReportHandler
from collections import defaultdict

def main():  
  # load JSON file with hyperparameters
  with open("config_params.json") as f:
    data = json.load(f)
  # Check for hyperparameters and parameters validation before running
  try:
    validated = HyperparamValidator(**data)
    combhandler = HyperparamCombinator(validated)
    # All combinations of hyperparameters
    combinations = combhandler.generate_combinations()

    # Instanciate NodeHandler with the Neo4j graph database connection
    # and sampling graph and generate the Test Set 
    node_handler = NodeHandler()
    test_ids_names, test_ids_caracteristcs = node_handler.hold_and_remove_movies_sample()

    # Iterate through all combinations of hyperparameters
    # sampling the remaing nodes and generating the Train/Validation Set
    experiment_name = 1
    for combination in combinations:
        # Unique experiment ID based on timestamp and iteration
        timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        report_builder = ReportHandler(timestamp=timestamp)
        # Extract hyperparameters values from the current combination
        fasrp_params = {k: v for k, v in combination.items() if k != "method"}
        len_hops = len(combination['iterationWeights'])
        search_methods = combination['method']
        # Hold validation set to evaluate current hyperparameters combination
        val_ids_names, val_ids_caracteristcs = node_handler.hold_and_remove_movies_sample()

        # Create embeddings for all User nodes remaining in the graph
        # using the current hyperparameters combination
        embedding_handler = UserEmbeddingHandler(fasrp_params)
        user_vectors_array = embedding_handler.create_user_vectors_array()
        evaluations = []
        for node in val_ids_names:
            node_handler.recreate_movie_nodes(node)
            node_handler.recreate_movie_attribute_rels(node, test_ids_caracteristcs[node["movieId"]])
            sub_graph_handler = NodeSubgraphHandler(node["movieId"], len_hops)
            node_subgraph_projection = sub_graph_handler.create_node_subgraph_projection()
            node_embedding_handler = ItemEmbeddingHandler(
                node["movieId"], node_subgraph_projection, fasrp_params)
            node_array = node_embedding_handler.create_item_vectors_array()
            for method in search_methods:
                node_vec_retriever = VectorRetriever(node_array, user_vectors_array, method=method, length=50)
                rec_users = node_vec_retriever.retrieve_users()
                node_evaluation = EvaluationHandler(rec_users)
                real_users = node_evaluation.retrive_actual_users()
                for k in [10, 20, 50]:
                    get_metrics = node_evaluation.calculate_metrics(real_users)
                    # During iteration:
                    evaluations.append({
                        "cutoff": k,
                        "precision": get_metrics[0],
                        "ndcg": get_metrics[1],
                        "method": method,
                    })
    # Calculate average metrics for group of nodes, metod and cutoff
    grouped = defaultdict(lambda: defaultdict(list))
    for e in evaluations:
        grouped[e["method"]][e["cutoff"]].append(e)
    average_metrics = {}
    for method, cutoffs in grouped.items():
        average_metrics[method] = {}
        for cutoff, metrics in cutoffs.items():
            avg_precision = sum(m["precision"] for m in metrics) / len(metrics)
            avg_ndcg = sum(m["ndcg"] for m in metrics) / len(metrics)
            average_metrics[method][cutoff] = {
                "average_precision": avg_precision,
                "average_ndcg": avg_ndcg
            }
    # Save the report with the average metrics for each method and cutoff
    for method, cutoffs in average_metrics.items():
        for cutoff, metrics in cutoffs.items():
            # Prepare metrics dict in the expected format
            metrics_dict = {
                f"precision_at_k_{cutoff}": metrics["average_precision"],
                f"ndcg_at_k_{cutoff}": metrics["average_ndcg"]
            }
            # Salve o relatório
            report_builder.save_report(
                hyperparameters=fasrp_params,
                method=method,
                metrics=metrics_dict,
                exp_id=experiment_name
            )
    # Recreate validation nodes and its attributes and
    # relationships in the graph
    node_handler.recreate_movie_nodes(val_ids_names)
    node_handler.recreate_movie_attribute_rels(val_ids_caracteristcs)
    node_handler.recreate_user_movie_rels(
        [node["movieId"] for node in val_ids_names])
    experiment_name += 1

  # Exception handling for invalid parameters and hyperparameters
  except Exception as e:
      print(f"Validation error: {e}")

if __name__ == "__main__":
    sys.exit(main())