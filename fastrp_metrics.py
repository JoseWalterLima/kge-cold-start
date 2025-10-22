# Avaliar nós de teste, gerando métricas por nó, métrica e valor de k
import sys
import os
import json
import time
from params_parser import HyperparamValidator, HyperparamCombinator
from src.node_handler import NodeHandler, NodeSubgraphHandler
from src.embedding_handler import UserEmbeddingHandler, ItemEmbeddingHandler
from src.vector_search_handler import VectorRetriever
from src.metrics_handler import EvaluationHandler, ReportHandler
from collections import defaultdict
import pandas as pd

# load test movie ids
test_ids_path = "experiments/test_ids.json"
with open(test_ids_path, "r", encoding="utf-8") as f:
    movie_ids = [t["movieId"] if isinstance(
        t, dict) else t for t in json.load(f)]
    
# load FastRP best hyperparameters
fastrp_params_path = "best_fastrp_params.json"
with open(fastrp_params_path, "r", encoding="utf-8") as f:
    all_params = json.load(f)
    # Extract hyperparameters values from the current combination
    fasrp_params = {k: v for k, v in all_params.items() if k != "method"}
    len_hops = len(all_params['iterationWeights'])
    search_method = all_params['method']

node_handler = NodeHandler()
# extrair relações/características dos nós de teste antes de removê-los
movies_id_name, movie_id_caracteristcs = node_handler.extract_movie_nodes_relations(movie_ids)
# remover todos nós de teste do grafo
node_handler.delete_nodes_and_rels(movie_ids)

# Create embeddings for all User nodes remaining in the graph
# using the current hyperparameters combination
user_embedding_handler = UserEmbeddingHandler(fasrp_params)
user_vectors_array = user_embedding_handler.create_user_vectors_array()

# Gerar embeddings para todos os nós do conjunto de teste
# realizar busca vetorial e calcular métricas de avaliação

# Inicializar listas para armazenar métricas por cutoff
precision_at_10 = []
ndcg_at_10 = []
precision_at_20 = []
ndcg_at_20 = []
precision_at_50 = []
ndcg_at_50 = []

for node in movies_id_name:
    # Criar embedding do nó de filme da iteração atual
    node_handler.recreate_movie_nodes(node)
    node_handler.recreate_movie_attribute_rels(
                    movie_id_caracteristcs[movie_id_caracteristcs["movieId"] == node["movieId"]])
    sub_graph_handler = NodeSubgraphHandler(node["movieId"], len_hops)
    node_subgraph_projection = sub_graph_handler.create_node_subgraph_projection()
    node_embedding_handler = ItemEmbeddingHandler(
                    node_subgraph_projection, node["movieId"], fasrp_params)
    node_array = node_embedding_handler.create_item_vector_array()

    # Realizar busca vetorial para o nó de filme atual
    node_vec_retriever = VectorRetriever(node_array, user_vectors_array, method=search_method, length=50)
    rec_users = node_vec_retriever.retrieve_users()
    node_evaluation = EvaluationHandler(rec_users)
    real_users = node_evaluation.retrive_actual_users()
    
    # Calcular métricas para cada valor de k
    for k in [10, 20, 50]:
        get_metrics = node_evaluation.calculate_metrics(real_users, k)
        if k == 10:
            precision_at_10.append(get_metrics[0])
            ndcg_at_10.append(get_metrics[1])
        elif k == 20:
            precision_at_20.append(get_metrics[0])
            ndcg_at_20.append(get_metrics[1])
        elif k == 50:
            precision_at_50.append(get_metrics[0])
            ndcg_at_50.append(get_metrics[1])

    node_handler.delete_nodes_and_rels(node["movieId"])

# Salvar as listas em um arquivo JSON
output_metrics = {
    "precision_at_10": precision_at_10,
    "ndcg_at_10": ndcg_at_10,
    "precision_at_20": precision_at_20,
    "ndcg_at_20": ndcg_at_20,
    "precision_at_50": precision_at_50,
    "ndcg_at_50": ndcg_at_50
}

with open("experiments/fastrp_final_metrics.json", "w", encoding="utf-8") as f:
    json.dump(output_metrics, f, indent=4, ensure_ascii=False)