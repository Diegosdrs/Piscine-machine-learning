# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 13:14:00 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from TinyStatistician import TinyStatistician

if __name__ == "__main__":
    # Liste d'exemple donnée dans l'exercice
    a = [1, 42, 300, 10, 59]
    
    # Création d'un objet TinyStatistician
    ts = TinyStatistician()

    # Test des méthodes avec la liste 'a'
    print("Moyenne de a : ", ts.mean(a))          # Moyenne
    print("Médiane de a : ", ts.median(a))        # Médiane
    print("Quartiles de a : ", ts.quartiles(a))    # Quartiles
    print("Percentile 10 de a : ", ts.percentile(a, 10))  # Percentile 10
    print("Percentile 15 de a : ", ts.percentile(a, 15))  # Percentile 15
    print("Percentile 20 de a : ", ts.percentile(a, 20))  # Percentile 20
    print("Variance de a : ", ts.var(a))          # Variance
    print("Écart-type de a : ", ts.std(a))        # Écart-type