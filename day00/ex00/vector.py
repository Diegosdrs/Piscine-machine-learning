# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vector.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 12:55:15 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

class Vector:
    def __init__(self, data):
        if isinstance(data, list):
            self.values = data
            if len(data) == 1:
                self.shape = (1, len(data[0]))
            else:
                self.shape = (len(data), 1)
        elif isinstance(data, int):
            self.values = [[float(i)] for i in range(data)]
            self.shape = (data, 1)
        elif isinstance(data, tuple) and len(data) == 2:
            a, b = data
            if a >= b:
                raise ValueError("The first value of the range must be smaller than the second.")
            self.values = [[float(i)] for i in range(a, b)]
            self.shape = (b - a, 1)
        else:
            raise TypeError("Invalid input type for Vector")

    def T(self):
        if self.shape[0] == 1:  # Si c'est un vecteur ligne (1, n)
            transposed_values = [[self.values[0][i]] for i in range(self.shape[1])]
            return Vector(transposed_values)  # Devient un vecteur colonne (n, 1)
        
        elif self.shape[1] == 1:  # Si c'est un vecteur colonne (n, 1)
            transposed_values = [[self.values[i][0] for i in range(self.shape[0])]]
            return Vector(transposed_values)  # Devient un vecteur ligne (1, n)

    def dot(self, v: 'Vector'):
        # Le produit scalaire est défini seulement pour des vecteurs de mêmes dimensions
        if self.shape != v.shape:
            raise ValueError(f"Shapes must match for dot product. Got {self.shape} and {v.shape}.")
        
        # Calcul du produit scalaire
        dot_product = sum(self.values[i][0] * v.values[i][0] for i in range(self.shape[0]))
        return Vector([[dot_product]])  # Retourne un vecteur avec le produit scalaire

    def __add__(self, other):
        if isinstance(other, Vector):  # Vérifier que l’autre élément est un Vector
            if self.shape != other.shape:  # Vérifier que les formes sont identiques
                raise ValueError(f"Vectors must have the same shape! Got {self.shape} and {other.shape}.")
            new_values = []
            for i in range(self.shape[0]):
                row = []
                for j in range(self.shape[1]):
                    row.append(self.values[i][j] + other.values[i][j])
                new_values.append(row)
            return Vector(new_values)
        elif isinstance(other, (int, float)):
            new_values = []
            for i in range(self.shape[0]):
                row = [self.values[i][0] + other]
                new_values.append(row)
            return Vector(new_values)
        else:
            raise TypeError("Addition is only defined between two Vectors or a Vector and a scalar.")

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            new_values = []
            for i in range(self.shape[0]):
                row = [other + self.values[i][0]]
                new_values.append(row)
            return Vector(new_values)
        else:
            raise TypeError("Addition only supported with scalar on the left side")

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Multiplication is only defined between a Vector and a scalar.")
        new_values = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                row.append(self.values[i][j] * scalar)
            new_values.append(row)
        return Vector(new_values)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __str__(self):
        return f"Vector({self.values})"

    def __repr__(self):
        return f"Vector({repr(self.values)})"

""" 
    def __mul__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Multiplication is only defined between two Vectors.")
        if self.shape != other.shape:
            raise ValueError(f"Vectors must have the same shape! Got {self.shape} and {other.shape}.")
        new_value = []
        for i in range (self.shape[0])
            row = []
            for j in range (self.shape[1])
                row.append(self.values[i][j] * other.values[i][j])
            new_value.append(row)
        return Vector(new_value) """