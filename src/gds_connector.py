# Author: José Walter Mota
# 07/2025
"""
Establish and return a Graph Data Science client.
Reads config from config.yaml file.
"""
import os
import yaml
from graphdatascience import GraphDataScience

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def get_gds_connection() -> GraphDataScience:
    uri = config['neo4j']['uri']
    user = config['neo4j']['user']
    pwd = config['neo4j']['pwd']
    timeout = config['neo4j']['timeout']
    try:
        gds = GraphDataScience(
            uri,
            auth=(user, pwd),
            timeout=timeout
        )
        gds.run_cypher("RETURN 1")
        return gds

    except Exception as e:
        raise ConnectionError(f"Failed to connect to GDS: {e}")
