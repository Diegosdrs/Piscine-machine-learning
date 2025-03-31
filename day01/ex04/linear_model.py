# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_model.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/31 12:39:13 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #charger les donnees
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1,1)
    Yscore = np.array(data["Score"]).reshape(-1,1)

    # Initialisation des modèles avec différentes valeurs de thetas
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))

    # predictions
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)

    # Calcul du MSE
    print(MyLR.mse_(Yscore, Y_model1))
    # 57.60304285714282
    print(mean_squared_error(Yscore, Y_model1))
    # 57.603042857142825
    print(MyLR.mse_(Yscore, Y_model2))
    # 232.16344285714285
    print(mean_squared_error(Yscore, Y_model2))
    # 232.16344285714285

    # Tracé du graphique des prédictions
    plt.scatter(Xpill, Yscore, color='blue', label='Données réelles')
    plt.plot(Xpill, Y_model1, color='green', label='Hypothèse (θ1 = -8)')
    plt.xlabel("Quantité de pilules bleues (microgrammes)")
    plt.ylabel("Score de conduite de l'espace")
    plt.legend()
    plt.title("Score de conduite en fonction des pilules bleues")
    plt.show()

    # Définir les valeurs de theta0 et theta1 à tester
    theta0_values = [80, 85, 90, 95]  # Différentes valeurs de θ0
    theta1_values = np.linspace(-14, -4, 50)  # 50 valeurs entre -10 et 0 pour θ1

    # Initialiser la figure
    plt.figure(figsize=(10, 6))

    # Calculer J(theta) pour chaque combinaison (θ0, θ1)
    for theta0 in theta0_values:
        loss_values = []
        for theta1 in theta1_values:
            model = MyLR(np.array([[theta0], [theta1]]))  # Créer le modèle avec (θ0, θ1)
            Y_pred = model.predict_(Xpill)  # Prédire les valeurs
            loss = model.mse_(Yscore, Y_pred)  # Calculer la MSE
            loss_values.append(loss)  # Stocker la perte

        # Tracer la courbe pour ce theta0
        plt.plot(theta1_values, loss_values, label=f"θ0 = {theta0}")

    # Mise en forme du graphique
    plt.xlabel("θ1 values")
    plt.ylabel("Cost J(θ)")
    plt.title("Évolution de la fonction de coût en fonction de θ1")
    plt.legend()  # Afficher la légende (θ0 correspondants)
    plt.grid()

    # Définir les nouvelles limites des axes
    plt.xlim(-14, -4)  # Limite en abscisse
    plt.ylim(0, 140)   # Limite en ordonnée

    plt.show()
