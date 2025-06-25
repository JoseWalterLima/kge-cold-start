# Author: José Walter Mota
# 11/2024
"""
Recomendação de grupo de clientes mais aderentes para novos ativos/papéis
usando graph embeddings.

Este módulo implementa um sistema de recomendação baseado em grafos de
conhecimento para sugerir grupos de clientes mais aderentes para um novo ativo.
A solução utiliza um banco de dados Neo4j como graph database e aplica técnicas
avançadas de machine learning em grafos.

Principais características:
- Utiliza um knowledge graph armazenado no Neo4j.
- Cria uma projeção do knowledge graph via biblioteca GraphDataScience.
- Cria graph embeddings usando o algoritmo FastRP (Fast Random Projection).
- Busca por Nearest Neighbors usando similaridade de cosseno entre vetores.

Notas:
- Este módulo assume a existência de um banco de dados Neo4j contendo o
knowledge graph dos dados de investimentos do Banco Inter.
- A chamada do módulo assume o recebimento de 5 argumentos, que são:
  - nm_modalidade: Nome do novo ativo (não presente no knowledge graph).
  - tipo_ativo: Modalidade do ativo, entre Renda Fixa e Renda Variável.
  - rendimento_taxa: Tipo de rendimento e taxa de juros.
  - indexador_pct: Indexador e percentual aos quais o ativo está atrelado.
  - n_clientes: Quantidade de clientes recomendados (default 100)
"""

from src.functions import connect_to_neo4j, create_nodes_and_relationships, \
            create_graph_projection, create_embeddings, get_node_labels, \
            find_similar_users
import streamlit as st
import time
from graphdatascience import GraphDataScience
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# App Streamlit
def login_page():
    """
    Esta função cria uma interface de usuário para inserção de credenciais
    do Neo4j, tenta estabelecer uma conexão e gerencia o estado da sessão
    após uma conexão bem-sucedida.

    Returns:
        None

    Side Effects:
        - Atualiza st.session_state com 'gds' e 'logged_in' após login
        bem-sucedido.
        - Utiliza st.rerun() para atualizar a aplicação após o login.

    Componentes de UI:
        - Campos de entrada para URI, usuário e senha.
        - Botão "Conectar" para iniciar a tentativa de conexão.
        - Mensagens de feedback sobre o status da conexão.
    """
    st.title("Knowledge Graph Connection")
    uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    username = st.text_input("User", value="neo4j")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        gds = connect_to_neo4j(uri, username, password)
        if gds:
            st.success("Successfully connected!")
            gds.graph.drop('item_cold_start')
            # Salva as informações de conexão na sessão
            st.session_state['gds'] = gds
            st.session_state['logged_in'] = True
            time.sleep(1)
            # Atualizar a execução
            st.rerun()

def main_page():
    """
    Renderiza e gerencia a página principal da aplicação de recomendação de
    clientes para ofertas públicas. Cria uma interface de usuário para entrada
    de dados sobre uma oferta pública, executa o processo de recomendação de
    clientes e exibe os resultados.

    Funcionalidades:
    1. Configuração da página e layout.
    2. Campos de entrada para detalhes da oferta pública.
    3. Validação de entradas e controle de ativação do botão de execução.
    4. Execução do processo de recomendação.
    5. Exibição dos resultados em uma tabela.
    6. Opção para download dos resultados em formato CSV.
    7. Limpeza dos dados temporários no Neo4j.
    8. Opção para desconexão do servidor Neo4j.

    Notas:
    - A função assume que uma conexão com o Neo4j já foi estabelecida e
    armazenada em st.session_state['gds'].
    - Implementa medidas de segurança para prevenir a desconexão durante
    a execução das funções.
    - Utiliza e modifica 'st.session_state' para gerenciar o estado de login e
    a conexão com o Neo4j.
    """
    st.set_page_config(layout="wide")
    st.title("Movie Recommender")
    
    # data input
    movie_name = st.text_input(
        "Movie Name", placeholder = "enter movie name"
    )
    release_date = st.text_input(
        "Release Date", placeholder = "e.g.: Jan-2010"
    )
    movie_genre = st.multiselect(
        "Movie Genre", options=[
            'Action', 'Adventure',
            'Animation', 'Childrens',
            'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'Film-Noir',
            'Horror', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller',
            'War', 'Western'
        ], placeholder = "select all that apply"
    )
    nur = st.number_input(
            "Number of user to recommend", value=100
            )
    def is_empty(field):
        """
        Verifica se um campo de input está vazio.
        """
        return not field.strip() if isinstance(field, str) else not field
    # Verificar todas as condições de entradas de dados
    all_fields_filled = not (
        is_empty(movie_name) or
        is_empty(release_date) or
        is_empty(movie_genre)
    )
    # Controlar o estado do botão "Desconectar"
    disconnect_button_disabled = False
    # Ativação do botão e mensagens de aviso
    if all_fields_filled:
        if st.button("Run Recommendation", disabled=False):
            # Bloquear a desconexão com o servidor durante a execução
            disconnect_button_disabled = True
            # Define variáveis de execução
            gds = st.session_state['gds']
            params = {
                "movieTitle": movie_name,
                "releaseDate": release_date,
                "genreDesc": movie_genre
            }
            # Executa processo de embedding e recomendação
            with st.spinner("Creating nodes and relationships..."):
                # Insere o novo produto no grafo e recupe o id
                target_id = create_nodes_and_relationships(gds, params)
                # Cria projeção do knowledge graph (em memória)
            with st.spinner("Creating graph projection..."):
                G = create_graph_projection(gds)
            with st.spinner("Building node embeddings..."):
                # Cria nodes embeddings, usando algorítmo específico
                emb = create_embeddings(gds, G, 'fastrp')
            with st.spinner("Retrieve node labels..."):
                all_node_ids = emb['nodeId'].tolist()
                labels_df = get_node_labels(gds, all_node_ids)
                emb = emb.merge(
                    labels_df, how='left', on='nodeId'
                    )
            with st.spinner("Searching users..."):
                # Filtra a base de embeddings, mantendo apenas
                # os nós usuários e o nó item alvo
                emb = emb[
                    (emb.label == 'User') | (emb['nodeId'] == target_id)
                    ].reset_index(drop=True).copy()
                # Busca pelos vetores mais próximos
                users_ids = find_similar_users(emb, target_id, nur)
                emb = emb[
                    emb['nodeId'].isin(users_ids)
                    ].reset_index(drop=True).copy()
                # Buscar user_id nos nós
                emb['userId'] = \
                    emb['nodeId'].apply(
                        lambda id: gds.util.asNode(id)['userId']
                        )
                users_list = list(emb['userId'].values)
                result_df = pd.DataFrame(
                    data={'userId': users_list}
                    )
                # Deletar projeção do grafo
                gds.graph.drop('item_cold_start')
                # Display results
                st.subheader("Most Recommended Users")
                st.dataframe(result_df)
                # Opção de download do resultado
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download resultados como CSV",
                    data=csv,
                    file_name="rec_result.csv",
                    mime="text/csv",
                )
                # Deletar ativo alvo do grafo
                gds.run_cypher(
                    """
                    MATCH (n)
                    WHERE id(n) = $target_id
                    DETACH DELETE n
                    """,
                    params={'target_id': target_id})
            # Habilitar desconxão com servidor Neo4j
            disconnect_button_disabled = False
    else:
        st.button("Executar Recomendação", disabled=True)
        # Mensagens de aviso específicas para cada campo
        st.error("Existem campos sem preenchimento.")
    # Permitir a desconexão do servidor
    if st.button("Desconectar", disabled=disconnect_button_disabled):
        # Fecha a conexão com o Neo4j
        if 'gds' in st.session_state:
            st.session_state['gds'].close()
        # Limpa as variáveis de sessão
        st.session_state['logged_in'] = False
        del st.session_state['gds']
        st.rerun()

def main():
    """
    Função principal que gerencia o fluxo da aplicação Streamlit.
    Controla a exibição das páginas com base no estado de login do usuário.

    Comportamento:
    - Se o usuário não estiver logado, exibe a página de login.
    - Se o usuário estiver logado, exibe a página principal da aplicação.
    """
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        login_page()
    else:
        main_page()

if __name__ == "__main__":
    main()
