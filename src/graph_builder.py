# Author: José Walter Mota
# 02/2025

from graphdatascience import GraphDataScience
import getpass

# Connect to Neo4j Graphdatabase (local server)
password=getpass.getpass("Password to neo4j server: ")
gds=GraphDataScience(
    'bolt://localhost:7687',
    auth=('neo4j', password)
)

# Building indexes
index_user = "CREATE INDEX FOR (b:User) ON (b.userId);"
gds.run_cypher(index_user)

index_movie = "CREATE INDEX FOR (b:Movie) ON (b.movieId);"
gds.run_cypher(index_movie)

# Building nodes
query = """
LOAD CSV WITH HEADERS FROM
'file:///data/userNode.csv' AS row
CREATE (b:User {userId: row.userId})
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM
'file:///data/ageNode.csv' AS row
CREATE (b:Age {ageValue: row.age})
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM
'file:///data/genderNode.csv' AS row
CREATE (b:Gender {genderDesc: row.gender})
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM
'file:///data/occupationNode.csv' AS row
CREATE (b:Occupation {occupationDesc: row.occupation})
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM
'file:///data/zipcodeNode.csv' AS row
CREATE (b:Zipcode {zipCode: row.zipcode})
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM
'file:///data/movieNode.csv' AS row
CREATE (b:Movie {movieId: row.movieId, movieTitle: row.movieTitle})
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM
'file:///data/genreNode.csv' AS row
CREATE (b:Genre {genreDesc: row.genreDesc})
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM
'file:///data/releaseNode.csv' AS row
CREATE (b:Release {releaseDate: row.releaseDate})
"""
gds.run_cypher(query)

# Building relationships
query = """
LOAD CSV WITH HEADERS FROM 'file:///data/watchedRel.csv' AS row
MATCH (a:User {userId: row.userId})
MATCH (b:Movie {movieId: row.movieId})
CREATE (a)-[:WATCHED]->(b)
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM 'file:///data/genreRel.csv' AS row
MATCH (a:Movie {movieId: row.movieId})
MATCH (b:Genre {genreDesc: row.genreDesc})
CREATE (a)-[:LABELED]->(b)
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM 'file:///data/releaseRel.csv' AS row
MATCH (a:Movie {movieId: row.movieId})
MATCH (b:Release {releaseDate: row.releaseDate})
CREATE (a)-[:RELEASED]->(b)
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM 'file:///data/ageRel.csv' AS row
MATCH (a:User {userId: row.userId})
MATCH (b:Age {ageValue: row.age})
CREATE (a)-[:HAS_AGE]->(b)
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM 'file:///data/genderRel.csv' AS row
MATCH (a:User {userId: row.userId})
MATCH (b:Gender {genderDesc: row.gender})
CREATE (a)-[:HAS_GENDER]->(b)
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM 'file:///data/occupationRel.csv' AS row
MATCH (a:User {userId: row.userId})
MATCH (b:Occupation {occupationDesc: row.occupation})
CREATE (a)-[:OCCUPATION]->(b)
"""
gds.run_cypher(query)

query = """
LOAD CSV WITH HEADERS FROM 'file:///data/residesRel.csv' AS row
MATCH (a:User {userId: row.userId})
MATCH (b:Zipcode {zipCode: row.zipcode})
CREATE (a)-[:LIVES_IN]->(b)
"""
gds.run_cypher(query)
