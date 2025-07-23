# Author: José Walter Mota
# 07/2025
"""
Establish and return a Graph Data Science client.
Reads config from config.yaml file.
"""
import os
import yaml
from graphdatascience import GraphDataScience

with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('src/pwd.yaml', 'r') as f:
    configpass = yaml.safe_load(f)

def get_gds_connection() -> GraphDataScience:
    uri = config['neo4j']['uri']
    user = config['neo4j']['user']
    pwd = configpass['neo4j']['pwd']
    try:
        gds = GraphDataScience(
            uri,
            auth=(user, pwd),
        )
        gds.run_cypher("RETURN 1")
        return gds

    except Exception as e:
        raise ConnectionError(f"Failed to connect to GDS: {e}")
