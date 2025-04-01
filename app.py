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
import yaml
from graphdatascience import GraphDataScience
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

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
    st.title("Login servidor Neo4j")
    uri = st.text_input("URI do Neo4j", value="bolt://localhost:7687")
    username = st.text_input("Usuário", value="neo4j")
    password = st.text_input("Senha", type="password")
    if st.button("Conectar"):
        gds = connect_to_neo4j(uri, username, password)
        if gds:
            st.success("Conexão bem-sucedida!")
            gds.graph.drop('item_cold_start')
            # Salva as informações de conexão na sessão
            st.session_state['gds'] = gds
            st.session_state['logged_in'] = True
            # Salva o max_id do grafo
            query="""
            MATCH (n)
            return count (n)
            """
            st.session_state['max_id'] =\
                gds.run_cypher(query).values[0][0]
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
    st.title("Oferta Pública: Recomendação de Clientes")
    col1, col2, col3 = st.columns(3)
    # Entrada dos dados
    with col1:
        nm_modalidade = st.text_input(
            "Nome do Papel", placeholder = "digite o nome do papel"
        )
        tipo_ativo = st.selectbox(
            "Modalidade", [
                "", "RF Emissao Terceiros", "RF Emissao Propria"
                ]
        )
        tp_rendimento = st.selectbox(
            "Remuneração", options=["", "PRE", "POS"]
        )
    with col2:
        if tp_rendimento=='PRE':
            taxa_indexador = st.text_input(
                "Taxa de Juros",
                placeholder = "ex: 2.1"
            )
        else:
            taxa_indexador = st.text_input(
                "Indexador",
                placeholder = "ex: CDI CETIP 100.0"
            )
        imposto_renda = st.selectbox(
            "Imposto de Renda", options=["", "Isento", "Não Isento"]
        )
        prazo_vencimento = st.text_input(
            "Vencimento", placeholder = "prazo máximo da aplicação (dias)"
        )
    with col3:
        prazo_minimo = st.text_input(
            "Liquidez", placeholder = "prazo mínimo da aplicação (dias)"
        )
        vl_min_aplic = st.text_input(
            "Aporte mínimo", placeholder = "ex: 500"
        )
        nur = st.number_input(
            "Número de Usuários Recomendados", value=100
            )
    def is_empty(field):
        """
        Verifica se um campo de input está vazio.
        """
        return not field.strip() if isinstance(field, str) else not field
    # Verificar todas as condições de entradas de dados
    all_fields_filled = not (
        is_empty(nm_modalidade) or
        is_empty(tipo_ativo) or
        is_empty(tp_rendimento) or
        is_empty(taxa_indexador) or
        is_empty(imposto_renda) or
        is_empty(prazo_vencimento) or
        is_empty(prazo_minimo) or
        is_empty(vl_min_aplic)
    )
    # Controlar o estado do botão "Desconectar"
    disconnect_button_disabled = False
    # Ativação do botão e mensagens de aviso
    if all_fields_filled:
        if st.button("Executar Recomendação", disabled=False):
            # Bloquear a desconexão com o servidor durante a execução
            disconnect_button_disabled = True
            # Define variáveis de execução
            gds = st.session_state['gds']
            params = {
                "nm_modalidade": nm_modalidade,
                "tipo_ativo": tipo_ativo,
                "tp_rendimento": tp_rendimento,
                "taxa_indexador": taxa_indexador,
                "imposto_renda": imposto_renda,
                "prazo_vencimento": prazo_vencimento,
                "prazo_minimo": prazo_minimo,
                "vl_min_aplic": vl_min_aplic
            }
            # Executa processo de embedding e recomendação
            with st.spinner("Criando embeddings..."):
                create_nodes_and_relationships(gds, params)
                G = create_graph_projection(gds)
                emb = create_embeddings(gds, G)
            with st.spinner("Recuperando node labels..."):
                all_node_ids = emb['nodeId'].tolist()
                labels_df = get_node_labels(gds, all_node_ids)
                emb = emb.merge(
                    labels_df, how='left', on='nodeId'
                    )
            with st.spinner("Buscando por usuários..."):
                # Recuperar o valor de max_id da sessão anterior (Login)
                max_id = st.session_state['max_id']
                # Seleciona apenas os nós usuários e ativo alvo
                emb = emb[
                    (emb.label == 'Usuario') | (emb.nodeId == max_id)
                    ].reset_index(drop=True).copy()
                # Busca pelos vetores mais próximos
                users_ids = find_similar_users(emb, max_id, nur)
                emb = emb[
                    emb['nodeId'].isin(users_ids)
                    ].reset_index(drop=True).copy()
                # Buscar user_id nos nós
                emb['user_id'] = \
                    emb['nodeId'].apply(
                        lambda id: gds.util.asNode(id)['usuarioId']
                        )
                users_list = list(emb['user_id'].values)
                result_df = pd.DataFrame(
                    data={'user_id': users_list}
                    )
                com_df = pd.read_csv(
                    'data/usuario_info.csv',
                    dtype={
                        'user_id': str,
                        'nu_cpfcnpj_cliente_ds': str,
                        'ano_nascimento_ds': str,
                        'nu_celular_cliente_ds': str}
                )
                result_df = result_df.merge(
                    com_df,
                    how='left',
                    on='user_id'
                ).copy()
                # Display results
                st.subheader("Usuários Recomendados")
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
                    WHERE id(n) > $max_id
                    DETACH DELETE n
                    """,
                    params={'max_id': max_id - 1})
                # Deletar projeção do grafo
                gds.graph.drop('item_cold_start')
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
