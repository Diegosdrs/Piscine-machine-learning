# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/25 09:49:54 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def loss_elem_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape or y.ndim != 2 or y.shape[1] != 1:
        return None
    res = np.zeros(y.shape)
    for i in range (y.shape[0]):
        res[i, 0] = (y[i, 0] - y_hat[i, 0]) ** 2
    return res

def loss_(y, y_hat):
    J_elem = loss_elem_(y, y_hat)
    if J_elem is None:
        return None
    return np.sum(J_elem) / (2 * len(y))

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

if __name__ =="__main__":
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict_(x1, theta1)  # Fonction predict définie ailleurs
    #print(y_hat1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    # Exemple 1 :
    print(loss_elem_(y1, y_hat1))
    # Output :
    # array([[0.], [1], [4], [9], [16]])

    # Exemple 2 :
    print(loss_(y1, y_hat1))
    # Output :
    # 3.0

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array(np.array([[0.], [1.]]))
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
    # Example 3:
    print(loss_elem_(y2, y_hat2))
    print(loss_(y2, y_hat2))
    # Output:
    2.142857142857143
    # Example 4:
    print(loss_(y2, y2))
    # Output:
    0.0