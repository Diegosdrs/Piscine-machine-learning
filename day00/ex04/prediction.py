# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 16:04:07 by dsindres         ###   ########.fr        #
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


if __name__ == "__main__":
    # Exemple 1 :
    x = np.arange(1, 6)  # vecteur x : [1, 2, 3, 4, 5]
    theta1 = np.array([[5], [0]])  # vecteur theta : [5, 0]
    result = predict_(x, theta1)
    print(result)
    # Sortie : 
    # array([[5.], [5.], [5.], [5.], [5.]])

    # Exemple 2 :
    theta2 = np.array([[0], [1]])  # vecteur theta : [0, 1]
    result2 = predict_(x, theta2)
    print(result2)
    # Sortie : 
    # array([[1.], [2.], [3.], [4.], [5.]])

    # Example 3:
    theta3 = np.array([[5], [3]])
    res3 = predict_(x, theta3)
    print(res3)
    # Output:
    #array([[ 8.], [11.], [14.], [17.], [20.]])
    
    # Example 4:
    theta4 = np.array([[-3], [1]])
    res4 = predict_(x, theta4)
    print(res4)
    # Output:
    #array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])