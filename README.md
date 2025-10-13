# **Abordagem do problema de cold-start, em sistemas de recomendação, como uma tarefa de completude em grafo de conhecimento**

Este repositório contém a implementação do código da dissertação de mestrado intitulada: "Abordagem do problema de cold-start, em sistemas de recomendação, como uma tarefa de completude em grafo de conhecimento", do Programa de Pós-Gradução em Ciência da Computação da UFMG. Utilizando uma estrutura leve e eficiente, a solução alcançou resultados expressivos ao recomendar itens mesmo na ausência de interações históricas, superando as métricas de Precision e NDCG do algoritmo LightFM em 30% e 130% respectivamente.


ADICIONAR AQUI UMA REVISÃO DA PROPOSTA.

De forma esquemática a aplicação tem o seguinte funcionamento:
![](esquema.png)

## Fonte de Dados
EXPLICAR SOBRE O MOVIE LENS 100K

## Estrutura da Aplicação
**Ambiente de Execução**: A aplicação, bem coma todas as bibliotecas definidas em *requirements.txt*, foi construída sobre a seguinte estrutura:
- Python 3.10.7
- Neo4j Community 5.26.3
- Neo4j Graph Data Science 2.13.2
- OpenJDK 21

**Organização dos Arquivos**:
- **sql/investment_data.sql**: Consulta e estruturação dos dados de investimento e de usuários, disponíveis no Trino.

- **sql/employees_data.sql**: Busca informações sobre funcionários do Banco Inter e Coligadas.

**Processamento dos Dados e Construção do Graph DB**:
- **src/process_data.py**: Transformação de dados, tratamento de dados ausentes, remoção de lideranças do Banco Inter e coligadas.

- **src/split_data.py**: Divide os dados em bases deduplicadas (normalizadas) e com informações específicas de cada nó e conexão, tornando a escrita do grafo mais eficiente (arquitetura de referência).

- **src/graph_builder.py**: Realiza carga em um banco de dados em grafo Neo4j, a partir de dados segmentados de investimentos.


## Colaboradores do trabalho
Este projeto é uma construção conjunta das áreas de PD II e PD III.
