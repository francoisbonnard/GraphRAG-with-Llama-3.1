# [GraphRAG-with-Llama-3.1 ](https://www.youtube.com/watch?v=nkbyD4joa0A)

## Poetry 
poetry install

poetry add ipykernel
poetry shell

python -m ipykernel install --user --name=myFuckingKernel --display-name "Python (myFuckingKernel)"

jupyter kernelspec list

## Ollama

Download Ollama

    ollama --help

choose [this version](https://ollama.com/library/llama3.1) of llama

    ollama run llama3.1:8b

## neo4j image docker

    docker compose up

### plugin APOC

APOC (which stands for Awesome Procedures On Cypher) is a popular and widely-used plugin for Neo4j, the graph database. It extends the capabilities of Neo4j’s Cypher query language by adding a large collection of utility procedures and functions. These utilities make it easier to work with data, perform complex operations, and extend Neo4j’s functionality beyond what’s available out-of-the-box.
