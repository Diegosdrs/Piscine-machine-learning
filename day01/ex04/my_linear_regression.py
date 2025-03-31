# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/31 10:08:43 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class MyLinearRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
    
    def predict_(self, x):
        # Check if x and theta are numpy arrays and non-empty
        if not isinstance(x, np.ndarray) or not isinstance(self.thetas, np.ndarray):
            return None
        if x.size == 0 or self.thetas.size == 0:
            return None
        if self.thetas.shape != (2, 1):
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Add a column of 1's to x to form the matrix X'
        X_prime = np.c_[np.ones(x.shape[0]), x]

        # Calculate the predicted values y_hat
        y_hat = np.dot(X_prime, self.thetas)
        
        return y_hat

    def loss_elem_(self, y, y_hat):
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
        
    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
            return None
        return np.sum((y_hat - y) ** 2) / (2 * y.shape[0])

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.thetas, np.ndarray) or not isinstance (self.max_iter, int) or not isinstance (self.alpha, float):
            return None
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None
        if x.shape[0] != y.shape[0]:
            return None
        for i in range(self.max_iter):
            self.thetas = self.thetas - self.alpha * self.gradient(x, y)
        return self.thetas

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.thetas, np.ndarray):
            return None
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None
        if x.shape[0] != y.shape[0] or self.thetas.shape != (2, 1):
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Construction de X' en ajoutant une colonne de 1
        X_prime = np.c_[np.ones(x.shape[0]), x]

        # Nombre d'exemples
        m = x.shape[0]

        # Calcul du gradient vectorisé
        gradient = (1 / m) * (X_prime.T @ (X_prime @ self.thetas - y))

        # X_prime * theta [ou] y_hat * theta = PREDICTION
        # PREDICTION - y = ERREUR entre la vrai valeur et la prediction
        # X_prime.T est la transposée de X_prime

        return gradient

    @staticmethod
    def mse_(y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
            return None
        return np.sum((y_hat - y) ** 2) / (y.shape[0])

    @staticmethod
    def rmse_(y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
            return None
        mse = np.sum((y_hat - y) ** 2) / (y.shape[0])
        rmse = np.sqrt(mse)
        return rmse

    @staticmethod
    def mae_(y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
            return None
        sum_abs = np.abs(y_hat - y)
        return np.sum(sum_abs) / (y.shape[0])

    @staticmethod
    def r2score_(y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
            return None
        y_mean = np.mean(y)
        result = 1 - (np.sum((y_hat - y) ** 2)) / (np.sum((y - y_mean) ** 2))
        return result


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    lr1 = MyLinearRegression(np.array([[2], [0.7]]))  # Modèle avec des thetas initiaux

    y_hat = lr1.predict_(x)
    print(y_hat)
    # Sortie attendue :
    # array([[10.74695094],
    #        [17.05055804],
    #        [24.08691674],
    #        [36.24020866],
    #        [42.25621131]])

    print("\n")

    print(lr1.loss_elem_(y, y_hat))
    # Sortie attendue :
    # array([[710.45867381],
    #        [364.68645485],
    #        [469.96221651],
    #        [108.97553412],
    #        [299.37111101]])

    print("\n")

    print(lr1.loss_(y, y_hat))
    # Sortie attendue : 195.34539903032385

    print("\n")
    print("-----------")
    
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    
    lr2.fit_(x, y)  # On entraîne le modèle

    print("\n")

    print(lr2.thetas)
    # Sortie attendue :
    # array([[1.40709365],
    #        [1.1150909 ]])

    print("\n")

    y_hat = lr2.predict_(x)
    print(y_hat)
    # Sortie attendue :
    # array([[15.3408728 ],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])
