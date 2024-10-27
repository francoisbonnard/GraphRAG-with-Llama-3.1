- [GraphRAG-with-Llama-3.1 ](#graphrag-with-llama-31-)
  - [Poetry](#poetry)
  - [Ollama](#ollama)
  - [neo4j image docker](#neo4j-image-docker)
    - [plugin APOC](#plugin-apoc)
  - [Nvidia warning](#nvidia-warning)
    - [1. **Ollama n'est pas configuré pour utiliser le GPU**](#1-ollama-nest-pas-configuré-pour-utiliser-le-gpu)
      - [Solution :](#solution-)
    - [2. **Problèmes de drivers ou CUDA non installés**](#2-problèmes-de-drivers-ou-cuda-non-installés)
      - [Solution :](#solution--1)
    - [3. **TensorFlow ou PyTorch pas configurés pour GPU**](#3-tensorflow-ou-pytorch-pas-configurés-pour-gpu)
      - [Solution :](#solution--2)
    - [4. **Ollama et le modèle utilisé ne supportent pas le GPU**](#4-ollama-et-le-modèle-utilisé-ne-supportent-pas-le-gpu)
      - [Solution :](#solution--3)
    - [Conclusion :](#conclusion-)
  - [Error -\> `with_structured_output`](#error---with_structured_output)
    - [Contexte de l'erreur :](#contexte-de-lerreur-)
    - [Solution 1 : Vérifier si votre LLM prend en charge `with_structured_output`](#solution-1--vérifier-si-votre-llm-prend-en-charge-with_structured_output)
    - [Solution 2 : Implémenter une alternative pour structurer les sorties](#solution-2--implémenter-une-alternative-pour-structurer-les-sorties)
    - [Exemple :](#exemple-)
    - [Solution 3 : Utiliser des outils alternatifs (LangChain Tools)](#solution-3--utiliser-des-outils-alternatifs-langchain-tools)
    - [Conclusion :](#conclusion--1)


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

check the model is ready

    ollama pull llama3.1


## neo4j image docker

    docker compose up

Pour arrêter :

    docker compose stop

### plugin APOC

APOC (which stands for Awesome Procedures On Cypher) is a popular and widely-used plugin for Neo4j, the graph database. It extends the capabilities of Neo4j’s Cypher query language by adding a large collection of utility procedures and functions. These utilities make it easier to work with data, perform complex operations, and extend Neo4j’s functionality beyond what’s available out-of-the-box.

## Nvidia warning 

L'image que vous avez fournie montre que la carte graphique **NVIDIA GeForce RTX** n'est pas utilisée, avec une utilisation à **0%**, alors que vous exécutez des calculs dans **Jupyter Notebook**. Cela signifie probablement que le modèle **Ollama** (ou tout autre modèle de calcul intensif) n'est pas configuré pour utiliser le GPU NVIDIA, mais plutôt le CPU ou peut-être le GPU intégré (**Intel UHD Graphics** dans ce cas).

Voici quelques raisons possibles pour lesquelles le GPU NVIDIA n'est pas utilisé et comment vous pourriez y remédier :

### 1. **Ollama n'est pas configuré pour utiliser le GPU**
Le modèle **Ollama** que vous utilisez (comme `llama3.1`) peut ne pas être configuré pour tirer parti du GPU. Certaines implémentations de LLM (modèles de langage) utilisent uniquement le CPU par défaut, même si un GPU est disponible.

#### Solution :
Vérifiez si **Ollama** prend en charge l'utilisation du GPU pour le modèle **llama3.1** et comment l'activer. Si c'est supporté, il pourrait y avoir une option spécifique à inclure lors de l'initialisation du modèle pour utiliser le GPU NVIDIA.

### 2. **Problèmes de drivers ou CUDA non installés**
Votre NVIDIA RTX ne peut pas être utilisée si les **drivers** NVIDIA ou **CUDA** (la plateforme de calcul parallèle de NVIDIA) ne sont pas correctement installés ou configurés.

#### Solution :
1. **Installez ou mettez à jour les drivers NVIDIA** depuis le [site officiel](https://www.nvidia.com/Download/index.aspx).
2. **Installez CUDA** et **cuDNN** (si nécessaire). Vous pouvez suivre les instructions pour l'installation de CUDA sur la [page officielle](https://developer.nvidia.com/cuda-downloads).
3. Redémarrez ensuite votre système pour vous assurer que toutes les mises à jour sont prises en compte.

### 3. **TensorFlow ou PyTorch pas configurés pour GPU**
Si vous utilisez des bibliothèques comme **TensorFlow** ou **PyTorch** en interne avec **Ollama**, il est possible que ces bibliothèques soient configurées pour utiliser uniquement le CPU.

#### Solution :
1. Si vous utilisez **TensorFlow**, assurez-vous d'avoir installé la version qui prend en charge le GPU :
   ```bash
   pip install tensorflow-gpu
   ```
   Ensuite, vérifiez si TensorFlow détecte le GPU :
   ```python
   import tensorflow as tf
   print("GPUs disponibles : ", tf.config.list_physical_devices('GPU'))
   ```

2. Si vous utilisez **PyTorch**, vous pouvez vérifier si le GPU est disponible avec :
   ```python
   import torch
   print(torch.cuda.is_available())  # True si le GPU est disponible
   ```

Si cela retourne `False`, il est possible que **PyTorch** ne soit pas installé avec le support GPU. Vous pouvez réinstaller PyTorch avec CUDA pour corriger cela :
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### 4. **Ollama et le modèle utilisé ne supportent pas le GPU**
Enfin, il est possible que l'implémentation spécifique d'**Ollama** ne prenne pas en charge l'accélération par GPU pour ce modèle particulier. Certains modèles sont optimisés pour fonctionner sur CPU uniquement.

#### Solution :
Vérifiez la documentation d'**Ollama** pour voir si le modèle **llama3.1** est compatible avec le GPU et s'il y a des instructions pour l'activer.

### Conclusion :
L'utilisation du GPU NVIDIA n'est pas activée dans votre cas actuel. Vous pouvez vérifier la prise en charge du GPU dans **Ollama** et vous assurer que vos bibliothèques et drivers NVIDIA sont correctement installés. Cela devrait vous permettre de tirer pleinement parti de votre GPU pour accélérer les calculs.

Si vous avez besoin d'aide supplémentaire pour vérifier la configuration, n'hésitez pas à me le dire !

## Error -> `with_structured_output`

L'erreur que vous rencontrez est un **`NotImplementedError`** dans le cadre de l'utilisation de la méthode `with_structured_output` sur l'objet **LLM**. Cela signifie que la méthode **`with_structured_output`** n'a pas encore été implémentée pour le modèle de langage que vous utilisez dans votre code, ou bien cette fonctionnalité n'est pas encore supportée pour le modèle spécifique que vous avez.

### Contexte de l'erreur :
- Vous essayez de définir un schéma structuré avec **`BaseModel`** à l'aide de **`Pydantic`** pour extraire des entités comme les noms d'organisations et de personnes.
- Ensuite, vous souhaitez lier ce schéma structuré avec la méthode **`with_structured_output`** du modèle de langage **LLM**.

### Solution 1 : Vérifier si votre LLM prend en charge `with_structured_output`
Vérifiez la documentation du modèle de langage que vous utilisez pour voir s'il prend en charge cette méthode. Certains modèles peuvent ne pas encore prendre en charge cette fonctionnalité, surtout s'ils sont relativement nouveaux ou n'ont pas encore intégré certaines fonctionnalités avancées comme les sorties structurées.

### Solution 2 : Implémenter une alternative pour structurer les sorties
Si le modèle de langage que vous utilisez ne prend pas en charge `with_structured_output`, vous pouvez utiliser une approche alternative pour obtenir des informations structurées. Une option consiste à extraire manuellement les entités de la réponse du modèle et à les structurer via **Pydantic**. Voici un exemple :

1. **Appeler le modèle pour obtenir une réponse brute**.
2. **Extraire les entités à partir de la réponse** (par exemple en utilisant des expressions régulières ou une analyse de texte).
3. **Structurer la sortie avec Pydantic**.

### Exemple :

```python
from pydantic import BaseModel, Field
import re

# Définir le modèle Pydantic pour structurer les résultats
class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
                    "appear in the text",
    )

# Supposons que la réponse du modèle soit un texte brut
llm_response = """
In the meeting, John Doe and Jane Smith, who work for OpenAI, discussed new advancements.
"""

# Extraire les noms des entités à l'aide d'une expression régulière simple (à adapter selon votre contexte)
names = re.findall(r'[A-Z][a-z]+\s[A-Z][a-z]+', llm_response)

# Structurer les données avec le modèle Pydantic
entities = Entities(names=names)

# Afficher les entités extraites
print(entities)
```

### Solution 3 : Utiliser des outils alternatifs (LangChain Tools)
Si vous travaillez avec **LangChain** et que vous souhaitez utiliser des outils pour extraire des entités, vous pouvez envisager d'utiliser une approche qui ne repose pas sur `with_structured_output`. Vous pourriez utiliser des outils ou des chaînes comme **NER (Named Entity Recognition)**, qui sont déjà bien établis et peuvent être plus facilement liés à votre modèle.

### Conclusion :
L'erreur **`NotImplementedError`** signifie que la méthode que vous tentez d'utiliser n'est pas encore disponible pour votre modèle. En attendant une implémentation, vous pouvez soit essayer une autre approche pour structurer les données manuellement, soit explorer des modèles de LLM ou des frameworks alternatifs qui offrent cette fonctionnalité.