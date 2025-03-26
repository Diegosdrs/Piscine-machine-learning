# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 13:50:58 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
#from vector import Vector

def simple_predict(x, theta):
    # Vérifier si x et theta sont des tableaux numpy non vides et ont les bonnes dimensions
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if x.ndim != 1 or theta.ndim != 1 or theta.size != 2:
        return None
    
    # Calcul des prédictions y_hat
    y_hat = theta[0] + theta[1] * x
    
    return y_hat.astype(float)

if __name__ == "__main__":
    # Exemple 1 : theta = [5, 0]
    x = np.arange(1, 6)  # x = [1, 2, 3, 4, 5]
    theta1 = np.array([5, 0])
    print(simple_predict(x, theta1))
    # Output : [5., 5., 5., 5., 5.]

    # Exemple 2 : theta = [0, 1]
    theta2 = np.array([0, 1])
    print(simple_predict(x, theta2))
    # Output : [1., 2., 3., 4., 5.]

    # Exemple 3 : theta = [5, 3]
    theta3 = np.array([5, 3])
    print(simple_predict(x, theta3))
    # Output : [ 8., 11., 14., 17., 20.]

    # Exemple 4 : theta = [-3, 1]
    theta4 = np.array([-3, 1])
    print(simple_predict(x, theta4))
    # Output : [-2., -1., 0., 1., 2.]