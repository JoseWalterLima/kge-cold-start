# Author: José Walter Mota
# 07/2025
"""
Need to add a description here!
"""
import pandas as pd

class FastrpTuner:
    #!!TODO!!
    def __init__(self, gds, config):
        #!!TODO!!
        self.config = config
        self.gds = gds
        self.resultados = {}

    def run_search(self):
        #!!TODO!!
        '''
        Este será um método público que
        chamará todos os métodos privados
        e irá retornar o dicionários com
        os resultados ao final do processo
        '''

    def _movie_nodes_sampling(self, sample_ratio=0.05) -> list:
        """
        Randomly samples a fraction of Movie nodes
        and returns a list of their IDs.
        """
        record = self.gds.run_cypher(
            "MATCH (m:Movie) RETURN count(m) AS total"
        )[0]
        total = record["total"]
        limit = int(round(total * sample_ratio))
        sampling_cypher = """
            MATCH (m:Movie)
            WITH m
            ORDER BY rand()
            LIMIT $limit
            RETURN m.movieId AS id
            """
        results = self.gds.run_cypher(
            sampling_cypher,
            params={"limit": limit}
            )
        return [row["id"] for row in results]
    
    def _extract_movie_nodes_relations(self, ids):
        """
        Fetches Movie attribute relationships for given movie IDs and
        returns the deduplicated movies list and grouped relations.
        """
        cypher = """
        UNWIND $ids AS movieId
        MATCH (m:Movie { movieId: movieId })-[r]->(c)
        RETURN
        m.movieId AS movieId,
        m.movieTitle AS movieTitle,
        type(r) AS relType,
        labels(c)[0] AS nodeLabel,
        c.genreDesc AS genreDesc,
        c.releaseDate AS releaseDate
        """
        raw = self.gds.run_cypher(cypher, params={"ids": ids})
        df = pd.DataFrame(raw)
        df_melted = df.melt(
                            id_vars=["movieId", "movieTitle", "relType", "nodeLabel"],
                            value_vars=["genreDesc", "releaseDate"],
                            var_name="attributeType",
                            value_name="attributeValue"
                        ).dropna(subset=["attributeValue"])
        movies_dict = (
            df_melted[["movieId", "movieTitle"]]
            .drop_duplicates()
            .to_dict("records")
        )
        groups = df_melted.groupby(
            ["nodeLabel", "attributeType", "relType"],
            sort=False
        )
        return movies_dict, groups

    def _delete_nodes_and_rels(self, ids):
        """
        Deletes nodes and all associated relationships
        based on provided node IDs.
        """
        query = """
        UNWIND $ids AS id
        MATCH (m:Movie { movieId: id })
        DETACH DELETE m
        """
        self.gds.run_cypher(query, params={"ids": ids})