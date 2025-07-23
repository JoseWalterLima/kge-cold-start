# Author: José Walter Mota
# 07/2025
"""
Need to add a description here!
"""
from gds_connector import get_gds_connection

class EmbeddingHandler:
    """
    Encapsula operações de embeddings no Neo4j via GDS.
    """
    def __init__(self, node_projection, relationship_projection):
        self.gds = get_gds_connection()
        self.node_projection = node_projection
        self.relationship_projection = relationship_projection

    def create_user_vectors(self, fastrp_params):
        """
        Project the full graph and generate FastRP embeddings.
        """
        projection = self._full_graph_projection()
        embeddings = self._create_fastrp_embeddings(projection, fastrp_params)
        #!TODO!!
        # filtrar somente nodes usuários, mas mantendo o id original
        # para que seja possível comprar com os usuários reais do movie
        # avaliar se mantém em memória ou se salva em alguma pasta
        return embeddings
    
    def _full_graph_projection(self):
        graph_name = "full_graph_projection"
        projection, metadata = self.gds.graph.project(
            graph_name,
            self.node_projection,
            self.relationship_projection
        )
        return projection

    def _create_fastrp_embeddings(self, projection, params):
        try:
            return self.gds.fastRP.stream(projection, **params)
        except Exception as e:
            raise RuntimeError(f"FastRP failed: {e}")

    def find_similar_users(user_vectors, item_vector, n_neighbors):
        """
        Encontra os usuários mais similares a um ativo específico usando o
        algoritmo Nearest Neighbors. Esta função utiliza embeddings de nós
        para calcular a similaridade entre um ativo alvo e outros nós no grafo,
        retornando os IDs dos nós mais similares.

        Args:
            df (pd.DataFrame): DataFrame contendo os nós e seus embeddings.
                            Deve ter colunas 'nodeId' e 'embedding'.
            item_id (int): ID do nó alvo para o qual se busca similares.
            n_neighbors (int): Número de vizinhos similares a serem retornados.

        Returns:
            list: Lista de IDs dos n_neighbors nós mais similares ao nó alvo,
                excluindo o próprio nó alvo.
        """
        embeddings = np.stack(df['embedding'].values)
        nn = NearestNeighbors(
            n_neighbors=(n_neighbors + 1),
            metric='cosine',
            algorithm='brute'
        )
        nn.fit(embeddings)
        target_index = df.index[df['nodeId'] == item_id].tolist()[0]
        _, indices = nn.kneighbors(
            [embeddings[target_index]],
            n_neighbors=(n_neighbors + 1)
        )
        return df.iloc[indices[0]]['nodeId'].values[1:]
