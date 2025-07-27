import sys
import json
from params_parser import HyperparamValidator, HyperparamCombinator
from src.node_handler import NodeHandler, NodeSubgraphHandler
from src.embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler

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
          # PAREI NA FUNÇÃO 7 DA LÓGICA!!









  # Exception handling for invalid parameters and hyperparameters
  except Exception as e:
      print(f"Validation error: {e}")

if __name__ == "__main__":
    sys.exit(main())