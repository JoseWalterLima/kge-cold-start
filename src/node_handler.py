# Author: José Walter Mota
# 07/2025

from src.gds_connector import get_gds_connection
import pandas as pd

class NodeHandler:
    """
    GDSNodeHandler manages node-level operations on graphs
    projected via Neo4j Graph Data Science. It supports tasks
    like sampling, deletion, and reinsertion of nodes for
    analytical workflows and knowledge graph preparation.
    """
    def __init__(self):
        self.gds = get_gds_connection()

    def hold_and_remove_movies_sample(self, sample_ratio=0.05):
        ids = self.sampling_movie_nodes(sample_ratio)
        movies_id_name, movie_id_caracteristcs = self.extract_movie_nodes_relations(ids)
        self.delete_nodes_and_rels(ids)
        return movies_id_name, movie_id_caracteristcs

    def sampling_movie_nodes(self, sample_ratio=0.05):
        """
        Randomly samples a fraction of Movie nodes
        with at least 50 user connections and returns a list of their IDs.
        """
        total = self.gds.run_cypher(
            """
            MATCH (m:Movie)-[:WATCHED]-(:User)
            WITH m, count(*) AS user_count
            WHERE user_count >= 50
            RETURN count(m) AS total
            """
        )['total'][0]
        limit = int(round(total * sample_ratio))
        sampling_cypher = """
            MATCH (m:Movie)-[:WATCHED]-(:User)
            WITH m, count(*) AS user_count
            WHERE user_count >= 50
            WITH m ORDER BY rand()
            LIMIT $limit
            RETURN m.movieId AS id
            """
        results = self.gds.run_cypher(
            sampling_cypher,
            params={"limit": limit}
            )
        return [row for row in results["id"]]
    
    def extract_movie_nodes_relations(self, ids):
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
        movies_id_name = (
            df_melted[["movieId", "movieTitle"]]
            .drop_duplicates()
            .to_dict("records")
        )
        return movies_id_name, df_melted

    def delete_nodes_and_rels(self, ids):
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
    
    def recreate_movie_nodes(self, movies_id_name):
        """
        Recreates Movie nodes from a list of dicts, merging on movieId
        and setting movieTitle only on creation
        """
        query = """
        UNWIND $movies AS movie
        MERGE (m:Movie {movieId: movie.movieId})
        ON CREATE SET m.movieTitle = movie.movieTitle
        """
        self.gds.run_cypher(query, params={"movies": movies_id_name})

    def recreate_movie_attribute_rels(self, df_melted):
        """
        Batch_recreates Movie_attribute relationships
        based on grouped data.
        """
        movie_id_caracteristcs = df_melted.groupby(
            ["nodeLabel", "attributeType", "relType"],
            sort=False
        )
        for (label, prop, rel), group in movie_id_caracteristcs:
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

class NodeSubgraphHandler:
    """ 
    Handles specific node subgraph construction and projections. 
    """
    def __init__(self, movie_id, hops=2):
        self.gds = get_gds_connection()
        self.movie_id = movie_id
        self.hops = hops

    def create_node_subgraph_projection(self):
        """ 
        Creates a subgraph projection around a specific
        movie node and returns the projection object. It uses
        the same hops as the FastRP embedding configuration for
        the current iteration on the optimization process.
        """
        result = self.gds.run_cypher(f"""
        MATCH (n:Movie {{movieId: $movie_id}})
        WITH id(n) AS targetId
        OPTIONAL MATCH (n)-[*1..{self.hops}]-(m)
        WITH collect(DISTINCT id(m)) AS neighborIds, targetId
        WITH [targetId] + neighborIds AS allIds
        UNWIND allIds AS id
        RETURN DISTINCT id
        """, params={"movie_id": self.movie_id})

        node_ids = result["id"].dropna().unique().tolist()

        node_spec = """
            UNWIND $nodeIds AS id
            RETURN id
        """

        relationship_spec = """
            MATCH (n)-[r]-(m)
            WHERE id(n) IN $nodeIds AND id(m) IN $nodeIds
            RETURN id(n) AS source, id(m) AS target, type(r) AS type
        """

        # Projection of the Sub Graph
        self.gds.graph.drop('subgraph_projection', False)
        projection, metadata = self.gds.graph.project.cypher(
            "subgraph_projection",
            node_spec,
            relationship_spec,
            parameters={"nodeIds": node_ids}
        )
        return projection
