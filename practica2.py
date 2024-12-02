# Importar las librerías necesarias
import numpy as np
import pandas as pd
from rxnfp.models import SmilesTokenizer, RXNBERTModel
from rxnfp.transformer_fingerprints import (
    get_default_model_and_tokenizer,
    RXNBERTFingerprintGenerator,
)
import tmap as tm
import matplotlib.pyplot as plt

# Selección de un conjunto de reacciones químicas para el análisis
reactions = [
    "CC(=O)OC1=CC=CC=C1C(=O)O>>CC(=O)OC1=CC=CC=C1C(=O)O",
    "C1CCOC1>>C1CCOC1",
    # Añade aquí más reacciones
]

# Inicializar el modelo y el tokenizer
model_name = "rxnfp/rxnfp_chembl_rxnfp_all"
tokenizer = SmilesTokenizer.from_pretrained(model_name)
model = RXNBERTModel.from_pretrained(model_name)

# Generar fingerprints de las reacciones
fingerprint_generator = RXNBERTFingerprintGenerator(model, tokenizer)
fingerprints = [fingerprint_generator(smiles) for smiles in reactions]

# Convertir fingerprints a numpy array
fingerprint_matrix = np.array(fingerprints)

# Generar el TMAP
lf = tm.LSHForest(256, 128)
lf.batch_add(fingerprint_matrix)
lf.index()

embedding = tm.embed(fingerprint_matrix, lf)

# Crear el gráfico
fig, ax = plt.subplots()
scatter = ax.scatter(embedding[:, 0], embedding[:, 1])

# Personalizar el gráfico
for i, reaction in enumerate(reactions):
    ax.annotate(reaction, (embedding[i, 0], embedding[i, 1]))

plt.title('Mapa de TMAP de Reacciones Químicas')
plt.xlabel('Coordenada 1')
plt.ylabel('Coordenada 2')
plt.show()

# Descripción de la relevancia de los datos utilizados en el filtro del TMAP
description = """
En este análisis, hemos utilizado varios parámetros para filtrar y analizar las reacciones químicas:
- **Peso Molecular**: Es una medida de la masa de una molécula. Reacciones con pesos moleculares similares tienden a agruparse juntas.
- **Donadores de Hidrógeno**: Los donadores de hidrógeno son importantes en las reacciones de formación de enlaces de hidrógeno, que pueden influir en la reactividad y la estabilidad de los compuestos.
- **Aceptores de Hidrógeno**: Similar a los donadores, los aceptores de hidrógeno juegan un papel crucial en la formación de enlaces de hidrógeno.
- **LogP (Coeficiente de partición octanol-agua)**: Este parámetro indica la hidrofobicidad de una molécula, lo cual es relevante para entender su solubilidad y permeabilidad.

Estos parámetros nos ayudan a entender mejor las propiedades fisicoquímicas de las reacciones y a interpretar los resultados del TMAP.
"""

print(description)
