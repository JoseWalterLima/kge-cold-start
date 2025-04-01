# Author: José Walter Mota
# 12/2024
"""
Este módulo contém as principais funções para interação com um banco de
dados Neo4j, criação de projeções de grafo, geração de embeddings e busca
de similaridade entre nós.

Principais funcionalidades:
1. Validação de conexão a um banco de dados Neo4j.
2. Criar nós e relacionamentos em um graph database Neo4j existente.
3. Gerar projeções de grafo usando o Neo4j Graph Data Science (GDS).
4. Criar embeddings de nós usando o algoritmo FastRP.
5. Encontrar nós similares com base em embeddings usando o algoritmo NN.

Nota: Este módulo é projetado para ser usado em conjunto com uma aplicação
Streamlit e requer acesso a um servidor Neo4j.
"""
import streamlit as st
from graphdatascience import GraphDataScience
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Funções base da aplicação
def connect_to_neo4j(uri:str, login:str, password:str):
    """
    Estabelece uma conexão com um banco de dados Neo4j usando as credenciais
    fornecidas. Esta função tenta criar uma instância de GraphDataScience para
    se conectar a um banco de dados Neo4j. Ela também testa a conexão executando
    uma consulta Cypher simples.

    Args:
        uri (str): O URI do banco de dados Neo4j.
        login (str): O nome de usuário para autenticação.
        password (str): A senha para autenticação.

    Returns:
        GraphDataScience: Uma instância de GraphDataScience conectada ao banco
        de dados, se a conexão for bem-sucedida.
        None: Se a conexão falhar.

    Raises:
        Exception: Captura qualquer exceção que ocorra durante a tentativa de
        conexão e exibe uma mensagem de erro usando st.error().
    """
    try:
        gds = GraphDataScience(uri, auth=(login, password))
        gds.run_cypher("RETURN 1")
        return gds
    except Exception as e:
        st.error(f"Falha na conexão: {str(e)}")
        return None

def create_nodes_and_relationships(gds, params: dict):
    """
    Cria nós e relacionamentos no banco de dados Neo4j com base nos
    parâmetros fornecidos.

    Args:
        gds (GraphDataScience): Uma instância conectada de GraphDataScience
        para executar consultas no Neo4j.
        params (dict): Um dicionário contendo os parâmetros para criar os nós
        e relacionamentos.

    Returns:
        Neo4j graph database nós e relacionamentos.
    """
    if params["tp_rendimento"]=="PRE":
        # Escrita dos nós
        query = """
        MERGE (a:Ativo {ativoNome: $nm_modalidade})
        MERGE (m:Modalidade {modalidadeDesc: $tipo_ativo})
        MERGE (mc:Marcador {marcadorDesc: $tp_rendimento})
        MERGE (j:Juros {jurosDesc: $taxa_indexador})
        MERGE (i:Imposto {impostoDesc: $imposto_renda})
        MERGE (p:Prazo {prazoDesc: $prazo_vencimento})
        MERGE (r:Resgate {resgateDesc: $prazo_minimo})
        MERGE (v:Valor {minimoDesc: $vl_min_aplic})
        """
        gds.run_cypher(query, params=params)
        # Escrita dos relacionamentos
        query = """
        MATCH (a:Ativo {ativoNome: $nm_modalidade}),
        (m:Modalidade {modalidadeDesc: $tipo_ativo}),
        (mc:Marcador {marcadorDesc: $tp_rendimento}),
        (j:Juros {jurosDesc: $taxa_indexador}),
        (i:Imposto {impostoDesc: $imposto_renda}),
        (p:Prazo {prazoDesc: $prazo_vencimento}),
        (r:Resgate {resgateDesc: $prazo_minimo}),
        (v:Valor {minimoDesc: $vl_min_aplic})
        MERGE (a)-[:CLASSIFICADO]->(m)
        MERGE (a)-[:MARCADO]->(mc)
        MERGE (a)-[:REMUNERADO]->(j)
        MERGE (a)-[:TRIBUTADO]->(i)
        MERGE (a)-[:VENCIMENTO]->(p)
        MERGE (a)-[:LIQUIDEZ]->(r)
        MERGE (a)-[:VALOR_MINIMO]->(v)
        """
        gds.run_cypher(query, params=params)
    else:
        # Escrita dos nós
        query = """
        MERGE (a:Ativo {ativoNome: $nm_modalidade})
        MERGE (m:Modalidade {modalidadeDesc: $tipo_ativo})
        MERGE (mc:Marcador {marcadorDesc: $tp_rendimento})
        MERGE (x:Indexador {indexadorDesc: $taxa_indexador})
        MERGE (i:Imposto {impostoDesc: $imposto_renda})
        MERGE (p:Prazo {prazoDesc: $prazo_vencimento})
        MERGE (r:Resgate {resgateDesc: $prazo_minimo})
        MERGE (v:Valor {minimoDesc: $vl_min_aplic})
        """
        gds.run_cypher(query, params=params)
        # Escrita dos relacionamentos
        query = """
        MATCH (a:Ativo {ativoNome: $nm_modalidade}),
        (m:Modalidade {modalidadeDesc: $tipo_ativo}),
        (mc:Marcador {marcadorDesc: $tp_rendimento}),
        (x:Indexador {indexadorDesc: $taxa_indexador}),
        (i:Imposto {impostoDesc: $imposto_renda}),
        (p:Prazo {prazoDesc: $prazo_vencimento}),
        (r:Resgate {resgateDesc: $prazo_minimo}),
        (v:Valor {minimoDesc: $vl_min_aplic})
        MERGE (a)-[:CLASSIFICADO]->(m)
        MERGE (a)-[:MARCADO]->(mc)
        MERGE (a)-[:INDEXADO]->(x)
        MERGE (a)-[:TRIBUTADO]->(i)
        MERGE (a)-[:VENCIMENTO]->(p)
        MERGE (a)-[:LIQUIDEZ]->(r)
        MERGE (a)-[:VALOR_MINIMO]->(v)
        """
        gds.run_cypher(query, params=params)

def create_graph_projection(gds):
    """
    Cria uma projeção de um grafo Neo4j utilizando a conector Graph
    DataScience (GDS). Esta função define projeções para nós e relacionamentos
    e então cria uma projeção de grafo nomeada "item_cold_start" no Neo4j.

    Args:
        gds (GraphDataScience): Uma instância conectada de GraphDataScience
        para executar operações no Neo4j.

    Returns:
        Graph: Um objeto Graph representando a projeção criada no Neo4j.

    Raises:
        Pode levantar exceções relacionadas à criação de projeções de grafo,
        como problemas de memória, configurações inválidas ou erros de conexão.
    """
    # Nós considerados na projeção
    node_projection = {
        "Usuario": {"label": "Usuario"},
        "Comunidade": {"label": "Comunidade"},
        "Gerente": {"label": "Gerente"},
        "Perfil": {"label": "Perfil"},
        "Nascimento": {"label": "Nascimento"},
        "Trabalho": {"label": "Trabalho"},
        "Formacao": {"label": "Formacao"},
        "Renda": {"label": "Renda"},
        "Estado": {"label": "Estado"},
        "Segmento": {"label": "Segmento"},
        "Ativo": {"label": "Ativo"},
        "Modalidade": {"label": "Modalidade"},
        "Marcador": {"label": "Marcador"},
        "Indexador": {"label": "Indexador"},
        "Juros": {"label": "Juros"},
        "Imposto": {"label": "Imposto"},
        "Prazo": {"label": "Prazo"},
        "Resgate": {"label": "Resgate"},
        "Valor": {"label": "Valor"}
    }
    # Relacionamentos considerados na projeção
    relationship_projection = {
        "CLASSIFICADO": {"type": "CLASSIFICADO", "orientation": "UNDIRECTED"},
        "ESTUDOU": {"type": "ESTUDOU", "orientation": "UNDIRECTED"},
        "INDEXADO": {"type": "INDEXADO", "orientation": "UNDIRECTED"},
        "INVESTIU": {"type": "INVESTIU", "orientation": "UNDIRECTED"},
        "LIQUIDEZ": {"type": "LIQUIDEZ", "orientation": "UNDIRECTED"},
        "MARCADO": {"type": "MARCADO", "orientation": "UNDIRECTED"},
        "MORA": {"type": "MORA", "orientation": "UNDIRECTED"},
        "NASCEU": {"type": "NASCEU", "orientation": "UNDIRECTED"},
        "OPERA": {"type": "OPERA", "orientation": "UNDIRECTED"},
        "ORIENTADO": {"type": "ORIENTADO", "orientation": "UNDIRECTED"},
        "PARTICIPA": {"type": "PARTICIPA", "orientation": "UNDIRECTED"},
        "PERTENCE": {"type": "PERTENCE", "orientation": "UNDIRECTED"},
        "RECEBE": {"type": "RECEBE", "orientation": "UNDIRECTED"},
        "REMUNERADO": {"type": "REMUNERADO", "orientation": "UNDIRECTED"},
        "TRABALHA": {"type": "TRABALHA", "orientation": "UNDIRECTED"},
        "TRIBUTADO": {"type": "TRIBUTADO", "orientation": "UNDIRECTED"},
        "VALOR_MINIMO": {"type": "VALOR_MINIMO", "orientation": "UNDIRECTED"},
        "VENCIMENTO": {"type": "VENCIMENTO", "orientation": "UNDIRECTED"}
    }
    G, _ = gds.graph.project(
        "item_cold_start", node_projection, relationship_projection
        )
    return G

def create_embeddings(gds, G):
    """
    Esta função utiliza o algoritmo FastRP do Neo4j Graph Data Science para
    gerar embeddings de baixa dimensão para os nós do grafo. Os embeddings
    são úteis para várias tarefas de aprendizado de máquina, como classificação
    de nós, link prediction e recomendação.

    Args:
        gds (GraphDataScience): Uma instância conectada de GraphDataScience
        para executar operações no Neo4j.
        G (Graph): O objeto Graph representando a projeção do grafo no Neo4j.

    Returns:
        pandas.DataFrame: Um DataFrame contendo os embeddings gerados para
        cada nó. As colunas incluem o ID do nó e as dimensões do embedding.

    Raises:
        Pode levantar exceções relacionadas à execução do algoritmo FastRP,
        como problemas de memória ou configurações inválidas.
    """
    emb = gds.fastRP.stream(
        G,
        randomSeed=42,
        embeddingDimension=128,
        normalizationStrength=-0.3,
        iterationWeights=[0, 0.2, 0.8]
    )
    return emb

def get_node_labels(gds, node_ids: list):
    """
    Retorna o rótulo de um nó específico no grafo Neo4j.

    Args:
        gds (GraphDataScience): Instância conectada de GraphDataScience.
        node_ids (list): Lista com todos os IDs dos nós no grafo.

    Returns:
        str: O primeiro rótulo de cada nó do grafo.
    """
    query = """
    UNWIND $node_ids AS id
    MATCH (n)
    WHERE id(n) = id
    RETURN id(n) AS nodeId, labels(n)[0] AS label
    """
    return gds.run_cypher(query, {'node_ids': node_ids})

def find_similar_users(df: pd.DataFrame, item_id: int, n_neighbors: int):
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
