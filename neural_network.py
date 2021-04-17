# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np

# # Calculs basiques

x = np.random.rand(1, 5).astype(np.float64)

weight = np.ones((5, 2), dtype=np.float64)

weight

np.matmul(x, weight)

x[0, 0] * weight[0, 0] + x[0, 1] * weight[1, 0] + x[0, 2] * weight[2, 0] + x[0, 3] * weight[3, 0] + x[0, 4] * weight[4, 0] == np.matmul(x, weight)[0]

# # MLP Classique

# Nous avons besoin de définir quelques points avant de commencer cette partie :
# 1. Combien de couches ?
# 2. Combien de cellules par couche ?
# 3. Combien de cellules en sortie ?
# 4. Quelle fonction d’activation ?
# 5. La fonction de perte ?
#
# **Réponses :**
# 1. Commençons par un réseau simple à 2 couches.
# 2. Partons sur une base de 10, 10.
# 3. Partons sur un problème à 2 classes.
# 4. Prenons la tangente hyperbolique (tanh).
# 5. Crosse Entropie.

# Déclarons nos premiers paramètres.

input_size = 5
hidden_cells = [10, 10]
# activation = lambda x: x # On transmet la méthode (par le biais de sa signature)
activation = np.tanh
nb_class = 2


# ## Softmax

def softmax(x: np.array) -> np.array:
    return np.exp(x) / np.vstack(np.sum(np.exp(x), axis=1))


# ## Negative Likelihood Loss

def nll(predictions: np.array, targets: np.array, mode=None) -> np.array:
    # Targets 1D
    assert (len(targets.shape) == 1),"1D array only"
    if mode == "mean":
        return -np.log(predictions[list(range(len(targets))), targets.tolist()]).mean()
    elif mode == "sum":
        return -np.log(predictions[list(range(len(targets))), targets.tolist()]).sum()
    else:
        return -np.log(predictions[list(range(len(targets))), targets.tolist()])


# ## Forward

# Écrivons nous une fonction qui va nous permettre de passer une batch au travers du réseau.

def forward(model : dict, x : np.array, activation) -> np.array:
    # Nous partons du principe que notre modèle est ordonné
    all_keys = list(model.keys())
    for i in range(len(all_keys) - 1):
        x = np.matmul(x, model[all_keys[i]]['weight']) + model[all_keys[i]]['bias']
        x = activation(x)
    y = np.matmul(x, model[all_keys[-1]]['weight']) + model[all_keys[-1]]['bias']
    return y


# # Backward

# TODO : Ici mettre la description du calcul de backward

def backward(loss):
    pass


# # Exemple de modèle

# +
layers_dict = {} # Dictionnaire qui fera office de modèle

last_input_size = input_size
for i in range(len(hidden_cells)):
    layers_dict[i] = {}
    layers_dict[i]['weight'] = np.random.normal(size=(last_input_size, hidden_cells[i]))
    layers_dict[i]['bias'] = np.zeros((1, hidden_cells[i]))
    last_input_size = hidden_cells[i]
layers_dict[i+1] = {'weight': np.random.normal(size=(last_input_size, nb_class)), 'bias': np.zeros((1, nb_class))}
# -


X_sample = np.random.randint(low=0, high=6, size=(10, 5)).astype(np.float64)
y_sample = np.random.randint(low=0, high=2, size=(10,))

output = forward(layers_dict, X_sample, activation)

output

# On peut appliquer la softmax sur notre sortie afin d’avoir la probabilité d’appartenance à chaque classe.

predictions = softmax(output)

# On détermine la perte (negative likelyhood loss)

loss = nll(predictions, y_sample, mode="mean")

# **À lire** : https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76




