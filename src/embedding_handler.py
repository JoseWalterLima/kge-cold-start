# Author: José Walter Mota
# 07/2025

from gds_connector import get_gds_connection
import numpy as np

class UserEmbeddingHandler:
    """
    Encapsulates embedding operations in Neo4j via GDS.
    """
    def __init__(self, params):
        self.gds = get_gds_connection()
        self.params = params

    def create_user_vectors_array(self):
        """
        Project the full graph, generate FastRP embeddings,
        and return the user ids as a list and their embeddings
        as a NumPy array.
        """
        projection = self.full_graph_projection()
        embeddings = self.create_user_fastrp_embeddings(projection)
        user_ids = self.get_user_node_ids(embeddings)
        return self.create_user_vectors(embeddings, user_ids)
    
    def full_graph_projection(self):
        graph_name = "full_graph_projection"
        projection, metadata = self.gds.graph.project(
            graph_name,
            self.node_projection,
            self.relationship_projection
        )
        return projection

    def create_user_fastrp_embeddings(self, projection):
        try:
            return self.gds.fastRP.stream(
                projection, randomSeed=42, **self.params)
        except Exception as e:
            raise RuntimeError(f"FastRP failed: {e}")

    def get_user_node_ids(self, embedding_df):
        query = """
        UNWIND $node_ids AS id
        MATCH (n)
        WHERE id(n) = id AND 'User' IN labels(n)
        RETURN id(n) AS nodeId, n.userId AS userId
        """
        return self.gds.run_cypher(query, {'node_ids':
                            [id for id in embedding_df['nodeId']]})

    def create_user_vectors(self, dfembedding, dfids):
        """
        Creates arrays of user vectors and a list of their original user IDs.
        """
        dfjoin = dfembedding.merge(dfids, how='inner', on='nodeId'
                          )[['userId','embedding']]
        return [dfjoin["userId"].astype('int64').to_numpy(), \
            np.stack(dfjoin["embedding"].values)]

class ItemEmbeddingHandler:
    """
    Handles embedding operations for a specific item node in a subgraph.
    """
    def __init__(self, subgraph_projection, target_node_id, params):
        self.gds = get_gds_connection()
        self.subgraph_projection = subgraph_projection
        self.target_node_id = target_node_id
        self.params = params

    def create_item_vector_array(self):
        subgraph_vectors = self.create_item_fastrp_embedding()
        item_vector = self.filter_target_embedding(subgraph_vectors)
        item_id = self.get_item_node_id()
        return [np.array(item_id), np.array(item_vector)]

    def create_item_fastrp_embedding(self):
        try:
            return self.gds.fastRP.stream(
                self.subgraph_projection, randomSeed=42, **self.params)
        except Exception as e:
            raise RuntimeError(f"FastRP failed: {e}")

    def filter_target_embedding(self, embedding_df):
        """
        Filters the embedding DataFrame to return only the embedding for the target node.
        """
        filtered = embedding_df[embedding_df['nodeId'] == self.target_node_id]
        if filtered.empty:
            raise ValueError(f"No embedding found for nodeId {self.target_node_id}")
        return filtered.iloc[0]['embedding']
    
    def get_item_node_id(self):
        query = """
        UNWIND $node_id AS id
        MATCH (n)
        WHERE id(n) = id
        RETURN id(n) AS nodeId
        """
        return self.gds.run_cypher(query, {'node_id': self.target_node_id}
                                   )["nodeId"].iloc[0]

