from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

class VectorRetriever:
    """
    Implements vector search methods to find similar users
    for a given item vector (item cold start scenario).
    - inputs:
        - **item_array**: 2D numpy array of item vector
        - **users_array**: 2D numpy array of user vectors
        - **method**: string indicating the similarity metric to be used in the search.
                      Allowed methods: 'cosine', 'euclidean', 'combined', 'ann', and 'all'.
                      **ps**: consine, euclidean and combined methods are applied in a brute-force
                              manner while 'ann' uses approximate nearest neighbors for faster search.
        - **length**: number of users to retrieve (default 100)
    - output: 2D numpy array of item_id and ordered user_ids
    """
    def __init__(self, item_array, users_array, method='cosine', length=100):
        self.item_array = item_array
        self.users_array = users_array
        self.method = method
        self.length = min(length, len(users_array[0]))

    def retrieve_users(self):
        if self.method == 'cosine':
            ordered_user_ids = self._cosine_similarity()
        elif self.method == 'euclidean':
            ordered_user_ids = self._euclidean_distances()
        # elif self.method == 'combined':
        #     ordered_user_ids = self._combined(self.item_array, self.users_array)
        # elif self.method == 'ann':
        #     ordered_user_ids = self._ann_search(self.item_array, self.users_array)
        # elif self.method == 'all':
        #     ordered_user_ids = np.array([])
        #     # aplicar cada uma das buscas de similaridade
        #     # appendendo o resultado como um array numpy
        #     # junto dos demais resultados, acrescentando
        #     # a descrição do método utilizado (ex: 'cosine',
        #     # 'euclidean', etc.) para permitir identificação
        #     # no relatório que será gerado pela classe MetricsHandler
        return self.item_array[0], np.array(ordered_user_ids)

    def _cosine_similarity(self):
        similarities = cosine_similarity(
                self.users_array[1],
                self.item_array[1].reshape(1, -1)
                ).flatten()
        top_indices = np.argsort(similarities)[::-1][:self.length]
        return [self.users_array[0][i] for i in top_indices]

    def _euclidean_distances(self):
        distances = euclidean_distances(
            self.users_array[1],
            self.item_array[1].reshape(1, -1)
        ).flatten()
        top_indices = np.argsort(distances)[:self.length]
        return [self.users_array[0][i] for i in top_indices]

    # def _combined(self):
    #     """
    #     CRIAR UM MÉTODO PARA COMBINAR CONSINE SIMILARITY E EUCLIDEAN DISTANCE
    #     DE MODO A GERAR UMA MÉTRICA ÚNICA QUE CONSIDERE ASPECTOS DE DISTÂNCIA,
    #     DIREÇÃO E TAMANHO DOS VETORES AO MESMO TEMPO.
    #     Combines cosine and euclidean similarities.
    #     """

    # def _ann_search(self):
    #     """
    #     Uses approximate nearest neighbors for faster search.
    #     Returns ordered user ids based on ANN search.
    #     """
