class VectorRetriever:
    """
    Provides various vector search methods between user vectors
    and a specific item vector (item cold start scenario).
    """
    def __init__(self, item_array, users_array, method='cosine', length=100):
        self.item_array = item_array
        self.users_array = users_array
        self.method = method
        self.length = min(length, users_array.shape[0])



    # embeddings = np.stack(df['embedding'].values)
    # nn = NearestNeighbors(
    #     n_neighbors=(n_neighbors + 1),
    #     metric='cosine',
    #     algorithm='brute'
    # )
    # nn.fit(embeddings)
    # target_index = df.index[df['nodeId'] == item_id].tolist()[0]
    # _, indices = nn.kneighbors(
    #     [embeddings[target_index]],
    #     n_neighbors=(n_neighbors + 1)
    # )
    # return df.iloc[indices[0]]['nodeId'].values[1:]
