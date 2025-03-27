# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/27 11:58:45 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def predict_(x, theta):
    # Check if x and theta are numpy arrays and non-empty
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if theta.shape != (2, 1):
        return None
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Add a column of 1's to x to form the matrix X'
    X_prime = np.c_[np.ones(x.shape[0]), x]

    # Calculate the predicted values y_hat
    y_hat = np.dot(X_prime, theta)
    
    return y_hat

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

def fit_(x, y, theta, alpha, max_iter):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray) or not isinstance (max_iter, int) or not isinstance (alpha, float):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    for i in range(max_iter):
        theta = theta - alpha * gradient(x, y, theta)
    return theta
    
    
if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    theta = np.array([1, 1]).reshape((-1, 1))

    # Exemple 0 :
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    # Output attendu :
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Exemple 1 :
    predictions = predict_(x, theta1)
    print(predictions)
    # Output attendu :
    # array([[15.3408728 ],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])