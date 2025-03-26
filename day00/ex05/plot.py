# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 15:29:03 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.ndim != 1 or y.ndim != 1 or theta.shape != (2, 1):
        return None
    if x.shape[0] != y.shape[0]:
        return None
    
    # Tracé des points de données
    plt.scatter(x, y, color='blue', label='Données réelles')
    
    # Calcul de la droite de régression
    y_hat = theta[0, 0] + theta[1, 0] * x
    
    # Tracé de la droite de régression
    plt.plot(x, y_hat, color='red', label='Régression linéaire')
    
    # Ajout des labels et du titre
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Régression Linéaire")
    plt.legend()
    
    # Affichage du graphique
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

    # Exemple 1 :
    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    # Exemple 2 :
    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    # Exemple 3 :
    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)