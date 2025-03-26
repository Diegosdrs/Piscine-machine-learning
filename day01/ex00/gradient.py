# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    gradient.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/26 10:51:44 by dsindres         ###   ########.fr        #
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

def simple_gradient(x, y, theta):
    if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if y.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    if x.shape[0] != y.shape[0]:
        return None

    # Utilisation de la fonction predict_
    y_hat = predict_(x, theta)
    if y_hat is None:
        return None

    # Calcul du gradient
    m = x.shape[0]
    error = y_hat - y
    grad_0 = np.sum(error) / m
    grad_1 = np.sum(error * x) / m

    return np.array([[grad_0], [grad_1]])


if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    theta2 = np.array([1, -0.4]).reshape((-1, 1))

    print("x:\n", x)
    print("\ny:\n", y)
    print("\ntheta1:\n", theta1)
    print("\nGradient:\n", simple_gradient(x, y, theta1))
    print(simple_gradient(x, y, theta2))