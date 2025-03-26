# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tools.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 14:21:17 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_intercept(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    # Ajouter une colonne de 1 à gauche du tableau
    # Si x est unidimensionnel (m * 1), on transforme d'abord en matrice (m * 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)  # Convertir en matrice colonne (m, 1)
    
    # Créer une matrice de 1s avec la même hauteur que x
    ones_column = np.ones((x.shape[0], 1))
    
    # Ajouter la colonne de 1s à gauche de x
    X = np.hstack((ones_column, x))
    
    return X
    

if __name__ == "__main__":
    x = np.arange(1,6)
    res = add_intercept(x)
    print(res)
    y = np.arange(1, 10).reshape((3, 3))
    res2 = add_intercept(y)
    print(res2)