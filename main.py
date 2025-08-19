import sys
import json
import time
from params_parser import HyperparamValidator, HyperparamCombinator
from src.node_handler import NodeHandler, NodeSubgraphHandler
from src.embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler
from src.vector_search_handler import VectorRetriever
from src.metrics_handler import EvaluationHandler, ReportHandler
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
        print("Test Sampling complete.")
        print(len(test_ids_names), "test nodes sampled.")
        
        # Unique experiment ID based on timestamp and iteration
        experiment_name = 1
        timestamp = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        report_builder = ReportHandler(timestamp=timestamp)

        # Iterate through all combinations of hyperparameters
        # sampling the remaing nodes and generating the Train/Validation Set
        for combination in combinations:
            # Extract hyperparameters values from the current combination
            fasrp_params = {k: v for k, v in combination.items() if k != "method"}
            len_hops = len(combination['iterationWeights'])
            search_methods = combination['method']
            # Sampling and hold validation set to evaluate current hyperparameters combination
            val_ids_names, val_ids_caracteristcs = node_handler.hold_and_remove_movies_sample()
            print("Validation Sampling complete.")
            print(len(val_ids_names), "validation nodes sampled.")

            # Create embeddings for all User nodes remaining in the graph
            # using the current hyperparameters combination
            embedding_handler = UserEmbeddingHandler(fasrp_params)
            user_vectors_array = embedding_handler.create_user_vectors_array()
            print("Embedding created for all users.")
            evaluations = []
            for node in val_ids_names:
                node_handler.recreate_movie_nodes(node)
                print(f"Node {node['movieId']} recreated.")
                node_handler.recreate_movie_attribute_rels(
                    val_ids_caracteristcs[val_ids_caracteristcs["movieId"] == node["movieId"]])
                print(f"Node {node['movieId']} attributes recreated.")
                sub_graph_handler = NodeSubgraphHandler(node["movieId"], len_hops)
                print(f"Subgraph for node {node['movieId']} created.")
                node_subgraph_projection = sub_graph_handler.create_node_subgraph_projection()
                print(f"Subgraph projection for node {node['movieId']} created.")
                node_embedding_handler = ItemEmbeddingHandler(
                    node_subgraph_projection, node["movieId"], fasrp_params)
                print(f"Embeddings for node {node['movieId']} created.")
                node_array = node_embedding_handler.create_item_vector_array()
                print(f"Array for node {node['movieId']} created.")
                # Iterate through all search methods for the current node
                for method in search_methods:
                    print(f"Running with method: {method}.")
                    node_vec_retriever = VectorRetriever(node_array, user_vectors_array, method=method, length=50)
                    print('Instantiate VectorRetriever.')
                    rec_users = node_vec_retriever.retrieve_users()
                    print('Retrieve users completed.')
                    node_evaluation = EvaluationHandler(rec_users)
                    print('Instantiate EvaluationHandler.')
                    real_users = node_evaluation.retrive_actual_users()
                    print("Actual users retrieved.")
                    for k in [10, 20, 50]:
                        get_metrics = node_evaluation.calculate_metrics(real_users, k)
                        # During iteration:
                        evaluations.append({
                            "cutoff": k,
                            "precision": get_metrics[0],
                            "ndcg": get_metrics[1],
                            "method": method,
                        })
                node_handler.delete_nodes_and_rels(node["movieId"])
                print(f"Node {node['movieId']} and its relationships deleted.")
            
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
                        "average_precision": round(avg_precision, 2),
                        "average_ndcg": round(avg_ndcg, 2)
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
            experiment_name += 1
            print(f"experiment_name: {experiment_name} completed.")
            print("First hiperparameters combination completed.")
            
            # Recreate validation nodes and its attributes and
            # relationships in the graph
            node_handler.recreate_movie_nodes(val_ids_names)
            node_handler.recreate_movie_attribute_rels(val_ids_caracteristcs)
            node_handler.recreate_user_movie_rels(
                [node["movieId"] for node in val_ids_names])
            print("Validation nodes and relationships recreated.")
        
        print("All hyperparameters combinations completed.")
        # Retrieve the best configuration for specific metric and k
        best_config = report_builder.get_best_config(metric='precision', k=50)
        print(f"Best configuration for precision at k=50: {best_config}")
        
        # Parse parameters for the best configuration
        best_fasrp_params = best_config['hyperparams']
        best_method = best_config['retrieval_method']
        best_len_hops = len(fasrp_params['iterationWeights'])
        validation_best_metrics = best_config['metrics']
        # Save the best configruration identified in the validation set
        report_builder.save_report(
            hyperparameters=best_fasrp_params,
            method=best_method,
            metrics=validation_best_metrics,
            exp_id="Best Validation Configuration"
        )

        # Create embeddings for all User nodes remaining in the graph
        # using the current hyperparameters combination
        embedding_handler = UserEmbeddingHandler(best_fasrp_params)
        user_vectors_array = embedding_handler.create_user_vectors_array()
        print("Embedding created for all users.")
        
        # Using the best configuration on the Test Set
        evaluations_test = []
        for node in test_ids_names:
            node_handler.recreate_movie_nodes(node)
            print(f"Node {node['movieId']} recreated.")
            node_handler.recreate_movie_attribute_rels(
                test_ids_caracteristcs[test_ids_caracteristcs["movieId"] == node["movieId"]])
            print(f"Node {node['movieId']} attributes recreated.")
            sub_graph_handler = NodeSubgraphHandler(node["movieId"], best_len_hops)
            print(f"Subgraph for node {node['movieId']} created.")
            node_subgraph_projection = sub_graph_handler.create_node_subgraph_projection()
            print(f"Subgraph projection for node {node['movieId']} created.")
            node_embedding_handler = ItemEmbeddingHandler(
                node_subgraph_projection, node["movieId"], best_fasrp_params)
            print(f"Embeddings for node {node['movieId']} created.")
            node_array = node_embedding_handler.create_item_vector_array()
            print(f"Array for node {node['movieId']} created.")

            # Search users using the best method and parameters
            print(f"Running with method: {best_method}.")
            node_vec_retriever = VectorRetriever(node_array, user_vectors_array, method=best_method, length=50)
            print('Instantiate VectorRetriever.')
            rec_users = node_vec_retriever.retrieve_users()
            print('Retrieve users completed.')
            node_evaluation = EvaluationHandler(rec_users)
            print('Instantiate EvaluationHandler.')
            real_users = node_evaluation.retrive_actual_users()
            print("Actual users retrieved.")
            for k in [10, 20, 50]:
                get_metrics = node_evaluation.calculate_metrics(real_users, k)
                # During iteration:
                evaluations_test.append({
                    "cutoff": k,
                    "precision": get_metrics[0],
                    "ndcg": get_metrics[1],
                    "method": best_method,
                })
            node_handler.delete_nodes_and_rels(node["movieId"])
            print(f"Node {node['movieId']} and its relationships deleted.")
        
        # Calculate average metrics for group of nodes, metod and cutoff
        grouped = defaultdict(lambda: defaultdict(list))
        for e in evaluations_test:
            grouped[e["method"]][e["cutoff"]].append(e)
        average_metrics = {}
        for method, cutoffs in grouped.items():
            average_metrics[method] = {}
            for cutoff, metrics in cutoffs.items():
                avg_precision = sum(m["precision"] for m in metrics) / len(metrics)
                avg_ndcg = sum(m["ndcg"] for m in metrics) / len(metrics)
                average_metrics[method][cutoff] = {
                    "average_precision": round(avg_precision, 2),
                    "average_ndcg": round(avg_ndcg, 2)
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
                    hyperparameters=best_fasrp_params,
                    method=method,
                    metrics=metrics_dict,
                    exp_id="Test Set"
                )
        # Recreate Test nodes and its attributes and
        # relationships in the graph
        node_handler.recreate_movie_nodes(test_ids_names)
        node_handler.recreate_movie_attribute_rels(test_ids_caracteristcs)
        node_handler.recreate_user_movie_rels(
            [node["movieId"] for node in test_ids_names])
        print("Test nodes and relationships recreated.")



    # Exception handling for invalid parameters and hyperparameters
    except Exception as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    sys.exit(main())