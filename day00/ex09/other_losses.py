# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    other_losses.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/25 11:23:46 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def mse_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    return np.sum((y_hat - y) ** 2) / (y.shape[0])

def rmse_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    mse = np.sum((y_hat - y) ** 2) / (y.shape[0])
    rmse = np.sqrt(mse)
    return rmse

def mae_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    sum_abs = np.abs(y_hat - y)
    return np.sum(sum_abs) / (y.shape[0])

def r2score_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    y_mean = np.mean(y)
    result = 1 - (np.sum((y_hat - y) ** 2)) / (np.sum((y - y_mean) ** 2))
    return result


if __name__ == "__main__":
    # Exemple 1:
    x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    # Erreur quadratique moyenne (MSE)
    print(mse_(x,y))
    # Sortie : 4.285714285714286

    # Erreur quadratique moyenne racine (RMSE)
    print(rmse_(x,y))
    # Sortie : 2.0701966780270626

    # Erreur absolue moyenne (MAE)
    print(mae_(x,y))
    # Sortie : 1.7142857142857142

    # R2 score
    print(r2score_(x,y))
    # Sortie : 0.9681721733858745





