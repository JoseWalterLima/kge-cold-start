import sys
import json
from params_parser import HyperparamValidator, HyperparamCombinator
from node_handler import NodeHandler, NodeSubgraphHandler
from embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler
from vector_search_handler import VectorRetriever
from metrics_handler import EvaluationHandler

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
                          "ndcg": get_metrics[1]
                      })
              # Calculate average precision at k for the current node
              precision_10 = [e["precision"] for e in evaluations if e["cutoff"] == 10]
              average_10 = sum(precision_10) / len(precision_10)
              ndcg_10 = [e["ndcg"] for e in evaluations if e["cutoff"] == 10]
              average_ndcg_10 = sum(ndcg_10) / len(ndcg_10)
              precision_20 = [e["precision"] for e in evaluations if e["cutoff"] == 20]
              average_20 = sum(precision_20) / len(precision_20)
              ndcg_20 = [e["ndcg"] for e in evaluations if e["cutoff"] == 20]
              average_ndcg_20 = sum(ndcg_20) / len(ndcg_20)
              precision_50 = [e["precision"] for e in evaluations if e["cutoff"] == 50]
              average_50 = sum(precision_50) / len(precision_50)
              ndcg_50 = [e["ndcg"] for e in evaluations if e["cutoff"] == 50]
              average_ndcg_50 = sum(ndcg_50) / len(ndcg_50)

  # Exception handling for invalid parameters and hyperparameters
  except Exception as e:
      print(f"Validation error: {e}")

if __name__ == "__main__":
    sys.exit(main())