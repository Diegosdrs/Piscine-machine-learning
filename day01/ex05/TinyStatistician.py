# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    TinyStatistician.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 13:16:28 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class TinyStatistician:
    def mean(self, data):
        m = len(data)
        if m == 0:
            return None
        if isinstance(data, np.ndarray):
            return np.mean(data)
        res = 0
        for i in data:
            res += i
        return res / m
        
    def median(self, data):
        m = len(data)
        if m == 0:
            return None
        # Si c'est un numpy.ndarray, trier avec sorted()
        if isinstance(data, np.ndarray):
            data_sorted = sorted(data)
        else:
            data_sorted = sorted(data)  # Tri pour les listes aussi
        # Si le nombre d'éléments est impair
        if m % 2 != 0:
            return data_sorted[(m - 1) // 2]
        # Si le nombre d'éléments est pair
        return (data_sorted[m // 2] + data_sorted[(m // 2) - 1]) / 2

    # def quartiles(self, data):
    #     m = len(data)
    #     if m == 0:
    #         return None
    #     if isinstance(data, np.ndarray):
    #         data_sorted = sorted(data)
    #     else:
    #         data_sorted = sorted(data)
    #     data_first = []
    #     data_second = []
    #     for i in data:
    #         if i <= self.median(data):
    #             data_first.append(i)
    #         else:
    #             data_second.append(i)
    #     Q1 = self.median(data_first)
    #     Q3 = self.median(data_second)
    #     return Q1, Q3

    def quartiles(self, x):
        if not isinstance(x, (list, np.ndarray)) or len(x) == 0:
            return None
        x_sorted = sorted(x)
        n = len(x_sorted)
        
        # Calcul du premier quartile (Q1)
        mid = n // 2
        if mid % 2 == 0:
            q1 = self.median(x_sorted[:mid])  # Médiane de la première moitié
        else:
            q1 = self.median(x_sorted[:mid+1])

        # Calcul du troisième quartile (Q3)
        if n % 2 == 0:
            q3 = self.median(x_sorted[mid:])  # Médiane de la seconde moitié
        else:
            q3 = self.median(x_sorted[mid+1:])

        return [q1, q3]

    def percentile(self, x, p):
        if not isinstance(x, (list, np.ndarray)) or len(x) == 0 or not isinstance(p, (int, float)):
            return None
        sorted_x = sorted(x)
        index = (p / 100) * (len(sorted_x) - 1)
        lower = sorted_x[int(index)]
        upper = sorted_x[min(int(index) + 1, len(sorted_x) - 1)]
        return round(lower + (upper - lower) * (index - int(index)), 1)

    def var(self, data):
        m = len(data)
        if m == 0:
            return None
        if isinstance(data, np.ndarray):
            data_sorted = sorted(data)
        else:
            data_sorted = sorted(data)
        moy = self.mean(data)
        res = 0
        for i in data:
            res += (i - moy) ** 2
        return res / m

    def std(self, data):
        m = len(data)
        if m == 0:
            return None
        if isinstance(data, np.ndarray):
            data_sorted = sorted(data)
        else:
            data_sorted = sorted(data)
        moy = self.mean(data)
        res = 0
        for i in data:
            res += (i - moy) ** 2
        res2 = ((res / m) ** 0.5)
        return res2
            
            
if __name__ == "__main__":
    tstat = TinyStatistician()

    # Test avec une liste
    a = [1, 42, 300, 10, 59]
    
    # Test de la fonction mean
    res = tstat.mean(a)
    print("Mean:", res)  # Expected result: 82.4
    
    # Test de la fonction median
    resu = tstat.median(a)
    print("Median:", resu)  # Expected result: 42.0
    
    # Test des quartiles
    q1, q3 = tstat.quartiles(a)
    print("Q1:", q1)  # Expected result: 10.0
    print("Q3:", q3)  # Expected result: 59.0
    
    # Test de la variance
    var_res = tstat.var(a)
    print("Variance:", var_res)  # Expected result: 3486.8 (approximatif)
    
    # Test de l'écart type
    std_res = tstat.std(a)
    print("Standard Deviation:", std_res)  # Expected result: 59.06 (approximatif)

    # Test avec un numpy.ndarray
    data = np.array([1, 42, 300, 10, 59])
    
    # Test de la fonction mean
    res = tstat.mean(data)
    print("Mean (numpy):", res)  # Expected result: 82.4
    
    # Test de la fonction median
    resu = tstat.median(data)
    print("Median (numpy):", resu)  # Expected result: 42.0
    
    # Test des quartiles
    q1, q3 = tstat.quartiles(data)
    print("Q1 (numpy):", q1)  # Expected result: 10.0
    print("Q3 (numpy):", q3)  # Expected result: 59.0
    
    # Test de la variance
    var_res = tstat.var(data)
    print("Variance (numpy):", var_res)  # Expected result: 3486.8 (approximatif)
    
    # Test de l'écart type
    std_res = tstat.std(data)
    print("Standard Deviation (numpy):", std_res)  # Expected result: 59.06 (approximatif)