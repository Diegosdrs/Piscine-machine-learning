# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_gradient.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/27 10:35:49 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape[0] != y.shape[0] or theta.shape != (2, 1):
        return None
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Construction de X' en ajoutant une colonne de 1
    X_prime = np.c_[np.ones(x.shape[0]), x]

    # Nombre d'exemples
    m = x.shape[0]

    # Calcul du gradient vectorisé
    gradient = (1 / m) * (X_prime.T @ (X_prime @ theta - y))

    # X_prime * theta [ou] y_hat * theta = PREDICTION
    # PREDICTION - y = ERREUR entre la vrai valeur et la prediction
    # X_prime.T est la transposée de X_prime

    return gradient
    

if __name__ == "__main__":
    # Exemple 0 :
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(gradient(x, y, theta1))
    # Sortie attendue :
    # array([[-19.0342...], [-586.6687...]])

    # Exemple 1 :
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(gradient(x, y, theta2))
    # Sortie attendue :
    # array([[-57.8682...], [-2230.1229...]])