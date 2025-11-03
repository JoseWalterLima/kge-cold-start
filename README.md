# **Cold-start como tarefa de completude em grafos de conhecimento: uma abordagem via encoders rasos de propagação.**

Este repositório contém a implementação do código da dissertação de mestrado intitulada: "Cold-start como tarefa de completude em grafos de conhecimento: uma abordagem via encoders rasos de propagação.", do Programa de Pós-Gradução em Ciência da Computação da UFMG. Utilizando uma estrutura leve e eficiente, a solução alcançou resultados expressivos ao recomendar itens mesmo na ausência de interações históricas, superando as métricas de Precision e NDCG em relação a um recomendador híbrido NDCG em relação a uma GNN indutiva.

## Fonte de Dados
Com o objetivo de garantir que o estudo de caso apresentado nesta dissertação seja publicamente verificável e reprodutível, optou-se pela utilização uma bases de dados públicas, amplamente reconhecidas e adotada em pesquisas de alta qualidade na área de sistemas de recomendação. Dentre as opções viáveis, foi dedicido pela utilização das bases de dados MovieLens 100k e MovieLens 1M. A ampla utilização destas bases de dados na literatura da área de sistemas de recomendação atesta sua qualidade nesse tipo de trabalho, além de possibilitar comparabilidade com estudos anteriores; as bases de dados preservam propriedades estruturais essenciais, como esparsidade e diversidade de perfis de usuários, o que torna a avaliação de métodos de recomendação representativa de cenários reais; o tamanho específico destas bases favorece reprodutibilidade e permite realizar experimentos mais extensos e robustos (ajuste de hiperparâmetros, validação cruzada e múltiplas repetições) sem custo computacional elevado; e, finalmente, a escolha facilita a comparação prática entre métodos com demandas computacionais distintas, assegurando que diferenças de desempenho reflitam méritos algorítmicos e não limitações de infraestrutura. Por fim, a utilização destas duas bases de dados permite a avaliação dos métodos em diferentes contextos de volume e diversidade de dados, possibilitando avaliações sobre escalabilidade e generalização dos resultados.

## Estrutura da Aplicação
**Ambiente de Execução**: A aplicação, bem coma todas as bibliotecas definidas em *requirements.txt*, foi construída sobre a seguinte estrutura:
- Python 3.10.7
- Neo4j Community 5.26.3
- Neo4j Graph Data Science 2.13.2
- OpenJDK 21

**Implementação do estudo de caso**
- **brach master**: Implementação das abordagens para a base de dados MovieLens 100k.
- **brach movielens-1M-implementation**: Implementação das abordagens para a base de dados MovieLens 1M.

**Processamento dos Dados e Construção do Graph DB**:
- **src/data_spliter.py**: Faz download dos dados e transforma a estrutura dos dados para otimizar o processo de construção do grafo de conhecimento.

- **src/graph_builder.py**: Realiza carga em um banco de dados em grafo Neo4j, a partir dos dados processados anteriormente.

**Implementação da completude em grafo de conhecimento**:
- **src/node_handler.py**: Classes para manipulação do nós do grafo, através de operações de escrita, deleção e atualização de nós e relacionamentos.

- **src/embedding_handler.py**: Classes para criação de embeddings dos nós do grafo, através de algoritmos implementados na lib Graph Data Science.

- **src/vector_search_handler.py**: Classes para implementação de busca vetorial através de diferentes métricas de similaridade.

- **src/metrics_handler.py**: Classes para calcular e reportar as métricas de eficiência e qualidade em ranqueamento: Hit rate@k, Precision@k e NDCG@k para diferentes valores de k.

**Implementação do algoritmo híbrido, via LightFM, para cold-strat de item**:
- **notebooks/lightfm_model.ipynb**: Implementação do algoritmo híbrido de recomendação para atuar como método convencional de resolução do problema de cold start, permitindo a comparação dos resultados de Hit rate@k, Precision@k e NDCG@K com o método proposto neste trabalho.

**Implementação da GNN indutiva (GraphSAGE) para cold-strat de item**:
- **notebooks/graphsage.ipynb.ipynb**: Implementação do algoritmo de Rede Neural em Grafo (GraphSAGE) para atuar como método estado-da-arte de resolução do problema de cold start, permitindo a comparação dos resultados de Hit rate@k, Precision@k e NDCG@K com o método proposto neste trabalho.

**Análises estatísticas e testes de hipóteses**
- **experiments/estatisticas_estudo_de_caso.ipynb**: Implementação do protocolo de avaliação offline utilizado para comparação entre os 3 métodos implementados neste trabalho. Foram feitas análises de amostras pareadas, de modo a possibilitar inferencias robustas e generalizáveis.


## Referência
Se você achou este trabalho útil, por favor considere citar:
```bibtex
@misc{cold-start-as-graph-completion,
      title={Cold-start como tarefa de completude em grafos de conhecimento: uma abordagem via encoders rasos de propagação.}, 
      author={José Walter de Lima Mota},
      year={2025},
      eprint={},
      archivePrefix={},
      primaryClass={},
      url={}, 
}
