# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_loss.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/25 10:26:12 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def loss_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    return np.sum((y_hat - y) ** 2) / (2 * y.shape[0])

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Exemple 1 :
    print(loss_(X, Y))
    # Output :
    # 2.142857142857143

    # Exemple 2 :
    print(loss_(X, X))
    # Output :
    # 0.0