# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    matrix.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 12:54:15 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from vector import Vector

class Matrix:
    def __init__(self, data):
        if isinstance(data, list):  # Initialisation avec liste de listes
            if not all(isinstance(row, list) for row in data):
                raise TypeError("Each row must be a list.")
            if not all(len(row) == len(data[0]) for row in data):
                raise ValueError("All rows must have the same number of columns.")
            self.values = data
            self.shape = (len(data), len(data[0]))
        elif isinstance(data, tuple) and len(data) == 2:  # Initialisation avec une shape (rows, cols)
            rows, cols = data
            if rows <= 0 or cols <= 0:
                raise ValueError("Matrix dimensions must be positive.")
            self.values = [[0.0] * cols for _ in range(rows)]
            self.shape = (rows, cols)
        else:
            raise TypeError("Invalid input for Matrix. Use list of lists or a (rows, cols) tuple.")
    
    def T(self):
        transposed = [[self.values[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
        return Matrix(transposed)
    
    def __add__(self, other):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Matrices must have the same shape for addition.")
        result = [[self.values[i][j] + other.values[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        return Matrix(result)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Matrices must have the same shape for subtraction.")
        result = [[self.values[i][j] - other.values[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        return Matrix(result)

    def __rsub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Subtraction only defined for Matrices.")
        # Soustraction inverse
        result = [[other.values[i][j] - self.values[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        return Matrix(result)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Multiplication par un scalaire
            result = [[self.values[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Matrix(result)
        elif isinstance(other, Matrix):  # Multiplication matricielle ou élément par élément
            if self.shape == other.shape:  # Produit d'Hadamard (élément par élément)
                result = [[self.values[i][j] * other.values[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Matrix(result)
            elif self.shape[1] == other.shape[0]:  # Produit matriciel classique
                result = [[sum(self.values[i][k] * other.values[k][j] for k in range(self.shape[1])) 
                        for j in range(other.shape[1])] for i in range(self.shape[0])]
                return Matrix(result)
            else:
                raise ValueError("Matrix multiplication requires matching inner dimensions or same shape for Hadamard product.")
        elif isinstance(other, Vector):  # Multiplication Matrice * Vecteur
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix-Vector multiplication requires the matrix's columns to match the vector's rows.")
            result = [[sum(self.values[i][k] * other.values[k][0] for k in range(self.shape[1]))] 
                      for i in range(self.shape[0])]
            return Vector(result)
        else:
            raise TypeError("Multiplication only supported with scalars, Matrices, or Vectors.")
    
    def __rmul__(self, other):
        return self.__mul__(other)  # La multiplication est commutative avec le scalaire

    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)) or scalar == 0:
            raise ValueError("Division requires a nonzero scalar.")
        result = [[self.values[i][j] / scalar for j in range(self.shape[1])] for i in range(self.shape[0])]
        return Matrix(result)

    def __rtruediv__(self, scalar):
        raise NotImplementedError("Division of Matrix by scalar is not defined in reverse direction.")

    def __str__(self):
        return '\n'.join(str(row) for row in self.values)
    
    def __repr__(self):
        return f"Matrix({self.values})"