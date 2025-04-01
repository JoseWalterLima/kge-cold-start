# **Addressing the item cold start problem as a knowledge graph completion task via graph embeding and vector search**

Esta aplicação implementa uma solução ao problema de cold start de item através da transposição deste em uma tarefa de knowledge graph completion.

Dentre as diversas abordagens possíveis neste contexto, optou-se pela criação de um [Knowledge Graph](https://neo4j.com/use-cases/knowledge-graph/?utm_source=GSearch&utm_medium=PaidSearch&utm_campaign=GenAI-KG-AMER-LATAM&utm_ID=&utm_term=knowledge%20graph&utm_adgroup=genai-specific_knowledge-graph&utm_content=-Sitetraffic--&utm_creative_format=Text&utm_marketing_tactic=SEMCO&utm_parent_camp=GenAI&utm_partner=na&utm_persona=SrDev&gad_source=1&gclid=EAIaIQobChMI_cn1_oXPiAMV4tXCBB3UJBpJEAAYASAAEgKP8vD_BwE), utilizando o banco de dados em grafo [Neo4j](https://neo4j.com/). Esta proposta visa a abstração do problema de negócio específico, permitindo a proposição de soluções para outros problemas de negócio relacionados, através de uma mesma estrutura de conhecimento. 

A aplicação utiliza o algoritmo [FastRP](https://arxiv.org/pdf/1908.11512) para criar embeddings dos nós do Knowledge Graph (preservando as informações de distância destes), de forma integrada ao algoritmo de aprendizado não-supervisionado [Nearest Neighbors](https://scikit-learn.org/1.5/modules/neighbors.html) para busca de vizinhos mais próximos (similares) no contexto de aplicação.

De forma esquemática a aplicação tem o seguinte funcionamento:
![](esquema.png)

## Fonte de Dados
Os dados de investimentos foram selecionados das bases existentes no Trino. Seguindo orientações das
áreas de negócio, e após diversas iterações, os dados foram selecionados das seguintes tabelas:

- **glue.analytics_invest.renda_fixa_obt_operacoes**: Informações sobre Renda Fixa Emissão Própria

- **glue.analytics_invest.report_operacao_unificada**: Informações sobre Renda Fixa Emissão Terceiros e Bolsa de Valores

- **glue.analytics_d2dbanking.conta_digital_obt_conta**: Informações gerais sobre os usuários e suas contas

- **glue.analytics_invest.suitability_clientes**: Perfil de Investidor (suitability)

- **glue.analytics_people_to_business_public.colaboradores_ativos**: Identificação dos colaboradores do Banco Inter e Coligadas

## Critérios de Seleção dos Dados
Levando em consideração diversas conversas com profissionais das áreas de negócios (investment
e business, principalmente), foram levantados diversos critérios para considerar um papel/ativo
como investimento, no contexto deste projeto, além de critérios para considerar os usuários/clientes
como investidores (pois nem todos os clientes são investidores). Estas medida é importante pois,
a qualidade de mapeamento do grafo está diretamente relacionada à qualidade dos dados (utilidade
da informação), no sentido de caracterizar o perfil dos usuários/clientes.
Foram selecionados todos os ativos que os clientes já investiram ao menos uma vez, a partir da
data de 01/01/2022. 

### Ativos/Papeis
Foram considerados como ativos os seguintes papeis (com suas características específicas):
- **Renda Fixa Emissão Própria**: nm_modalidade, tipo_ativo, tp_rendimento, vl_taxa,
  dc_indexador, pc_indexador_operacao, fl_isento_ir, nu_prazo_dias_corridos, nu_prazo_min_aplic,
  vl_min_aplic

- **Renda Fixa Emissão Terceiros**: nm_modalidade, tipo_ativo, tp_rendimento, vl_taxa, dc_indexador,        
  pc_indexador_operacao, nu_prazo_dias_corridos (na tabela "renda_fixa_terceiros_obt_posicao" não estão disponíveis as informações: fl_isento_ir, nu_prazo_min_aplic, vl_min_aplic)

- **Bolsa de Valores**: nm_modalidade, tipo_ativo (podem ser utilizadas outras características
    que auxiliem na qualificação destes papeis, mas este não foi o foco da primeira versão da
    aplicação)

Foram desconsiderados os ativos das seguintes categorias de **sg_papel**:
- Fundos, COE, TIME DEPOSIT, PREVIDENCIA, CRYPTO, CGI, APEX SALDO, APEX POSICAO,
  DI IMOB, CDIR PRE, CDI POS L, CDI POS, AP.AUT POS., TESOURO DIRETO 

### Usuários/Clientes
Foram considerados como investidores, objeto de recomendação de novos investimentos, 
os clientes de atenderam concomitantemente aos critérios:
- Conta ativa no Banco Inter, Tenha outros investimentos além da poupança, Não façam
parte da alta direção do Banco Inter ou Coligadas.

Foram consideradas as seguintes informações, para caracterizar o perfil dos usurários:
- dc_escolaridade, dc_profissao, vl_renda_mensal, dc_segmento, nm_gerente,
dc_sexo_ds, ano_nascimento_ds, ed_uf_cliente_ds

## Pontos de Melhoria da Aplicação
- **Posição em Carteira**: Foram selecionados todos os ativos os quais os usuários já investiram, ao menos
uma vez, no período de 01/01/2024 até 31/01/2025 (seguindo uma orientação da área de investimentos, por conta da qualidade dos dados). Entretanto, não foi possível avaliar quais ativos ainda estão
em aberto na carteira dos usuários e quais já estão encerrados. Esta é a abordagem padrão em sistemas de recomendação, não implicando em erros na solução, mas provavelmente para fins de negócio seria mais
produtivo ter visibilidade de quais ativos o usuário já teve e finalizaram e quais ainda estão em aberto.

- **Ativos correntes ou passados**: Diferenciar os ativos quem compõem a carteira dos investidores
com dois relacionamentos distintos: relacionamento **INVESTIU** (para ativos já liquidados ou vendidos)
e relacionamento **INVESTE** (para ativos que ainda estão dentro do pazo de validade ou com posição em
aberto na carteira do investidor).

- **Aportes**: Contabilizar o numero de aportes feitos em um mesmo ativo, bem como o valor total
aportado no ativo. Estas informações ajudariam a diferenciar a relevância dos ativos dentro de cateiras
similares para diferentes investidores.

## Estrutura da Aplicação
**Definições de Ambiente**: A aplicação, bem coma todas as bibliotecas definidas em *requirements.txt*, foi construída sobre a seguinte estrutura:
- Python 3.10.7
- Neo4j Community 5.26.3
- Neo4j Graph Data Science 2.15.0
- OpenJDK 21

**Busca dos Dados**:
- **sql/investment_data.sql**: Consulta e estruturação dos dados de investimento e de usuários, disponíveis no Trino.

- **sql/employees_data.sql**: Busca informações sobre funcionários do Banco Inter e Coligadas.

**Processamento dos Dados e Construção do Graph DB**:
- **src/process_data.py**: Transformação de dados, tratamento de dados ausentes, remoção de lideranças do Banco Inter e coligadas.

- **src/split_data.py**: Divide os dados em bases deduplicadas (normalizadas) e com informações específicas de cada nó e conexão, tornando a escrita do grafo mais eficiente (arquitetura de referência).

- **src/graph_builder.py**: Realiza carga em um banco de dados em grafo Neo4j, a partir de dados segmentados de investimentos.

**Aplicação Streamlit**:
- **src/functions.py**: Funções padrão para interação com banco de
dados Neo4j, criação de projeções de grafo, construção de embeddings e busca
de similaridade entre embeddings.
- **app.py**: Aplicação Streamlit (front‑end) de um sistema de recomendação para novas ofertas públicas.

## Colaboradores do trabalho
Este projeto é uma construção conjunta das áreas de PD II e PD III.
