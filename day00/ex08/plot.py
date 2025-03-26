# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/25 10:55:48 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

def loss_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    return np.sum((y_hat - y) ** 2) / (y.shape[0])

def plot_with_loss(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or y.size == 0 or theta.size != 2:
        return None

    # Prédictions
    y_hat = theta[0] + theta[1] * x
    
    # Calcul de la perte (J)
    J = loss_(y, y_hat)

    # Tracé des données et de la ligne de prédiction
    plt.scatter(x, y, color='blue', label='Données réelles')  # Tracer les points de données
    plt.plot(x, y_hat, color='orange', label='Ligne de prédiction')  # Tracer la ligne de prédiction
    
    # Tracer les lignes de perte entre chaque point et la prédiction
    for xi, yi, yhi in zip(x, y, y_hat):
        plt.plot([xi, xi], [yi, yhi], color='red', linestyle='--')  # Ligne entre chaque point et sa prédiction
    
    # Ajouter le titre avec la perte
    plt.title(f"Perte (J) = {J:.2f}")
    
    # Ajouter les labels
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Afficher la légende
    plt.legend()

    # Afficher le graphique
    plt.show()

if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

    # Exemple 1:
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)
    # Sortie : Figure X.1: Exemple 1

    # Exemple 2:
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)
    # Sortie : Figure X.2: Exemple 2

    # Exemple 3:
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
    # Sortie : Figure X.3: Exemple 3