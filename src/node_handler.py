# Author: José Walter Mota
# 07/2025
"""
Need to add a description here!
"""
import pandas as pd

class NodeHandler:
    #!!TODO!!
    def __init__(self, gds):#, config):
        #!!TODO!!
        #self.config = config
        self.gds = gds
        #self.resultados = {}

    def _sampling_movie_nodes(self, sample_ratio=0.05):
        """
        Randomly samples a fraction of Movie nodes
        and returns a list of their IDs.
        """
        total = self.gds.run_cypher(
            "MATCH (m:Movie) RETURN count(m) AS total"
        )['total'][0]
        limit = int(round(total * sample_ratio))
        sampling_cypher = """
            MATCH (m:Movie)
            WITH m ORDER BY rand()
            LIMIT $limit
            RETURN m.movieId AS id
            """
        results = self.gds.run_cypher(
            sampling_cypher,
            params={"limit": limit}
            )
        return [row for row in results["id"]]
    
    def _extract_movie_nodes_relations(self, ids):
        """
        Fetches Movie_attribute relationships for given movie IDs and
        returns the deduplicated movies list and grouped relations.
        Does not return User_Movie relationships.
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
    
    def _recreate_movie_nodes(self, movies_dict):
        """
        Recreates Movie nodes from a list of dicts, merging on movieId
        and setting movieTitle only on creation
        """
        query = """
        UNWIND $movies AS movie
        MERGE (m:Movie {movieId: movie.movieId})
        ON CREATE SET m.movieTitle = movie.movieTitle
        """
        self.gds.run_cypher(query, params={"movies": movies_dict})

    def _recreate_movie_attribute_rels(self, groups):
        """
        Batch_recreates Movie_attribute relationships
        based on grouped data.
        """
        for (label, prop, rel), group in groups:
            batch = (
                group[["movieId", "attributeValue"]]
                .rename(columns={"attributeValue": "value"})
                .to_dict("records")
            )
            cypher = f"""
            UNWIND $batch AS row
            MATCH (m:Movie {{ movieId: row.movieId }})
            MERGE (c:`{label}` {{ {prop}: row.value }})
            MERGE (m)-[:{rel}]->(c)
            """
            self.gds.run_cypher(cypher, params={"batch": batch})
    
    def recreate_user_movie_rels(self, movie_ids, csv_path="data/watchedRel.csv"):
        """
        Reads a CSV of User_Movie links, filters for the given movie IDs,
        and recreates WATCHED relationships in the Neo4j graph.
        """
        rels = pd.read_csv(csv_path,
                         dtype={'userId': str, 'movieId': str}
                         )
        rels = rels[rels["movieId"].isin(movie_ids)]
        rels = rels[['userId', 'movieId']].to_dict("records")
        cypher = """
        UNWIND $relations AS rel
        MATCH (u:User  { userId: rel.userId })
        MATCH (m:Movie { movieId: rel.movieId })
        MERGE (u)-[:WATCHED]->(m)
        """
        self.gds.run_cypher(cypher, params={"relations": rels})
