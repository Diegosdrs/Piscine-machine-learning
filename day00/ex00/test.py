# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/24 11:47:22 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from matrix import Matrix
from vector import Vector

def main():
    # --------------------------------------------------
    # 1. Création de matrices de test
    # --------------------------------------------------
    
    # Matrice 2x4
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    print("Matrice m1 (2x4) :")
    print(m1)
    
    # Matrice 4x2
    m2 = Matrix([[0.0, 1.0],
                 [2.0, 3.0],
                 [4.0, 5.0],
                 [6.0, 7.0]])
    print("\nMatrice m2 (4x2) :")
    print(m2)

    # --------------------------------------------------
    # 2. Test de la transposition (m1.T())
    # --------------------------------------------------
    print("\nTransposition de m1 :")
    print(m1.T())

    # --------------------------------------------------
    # 3. Test de l'addition de matrices (m1 + m1)
    # --------------------------------------------------
    print("\nAddition de m1 et m1 (m1 + m1) :")
    print(m1 + m1)

    # --------------------------------------------------
    # 4. Test de la multiplication de matrices (m1 * m2)
    # --------------------------------------------------
    print("\nMultiplication de m1 et m2 (m1 * m2) :")
    print(m1 * m2)

    # --------------------------------------------------
    # 5. Test de la multiplication matrice-vecteur (m1 * v2)
    # --------------------------------------------------
    v2 = Vector([[1.0], [2.0], [3.0], [4.0]])  # Vecteur colonne 4x1
    print("\nMultiplication de m1 et v2 (m1 * v2) :")
    print(m1 * v2)

    # --------------------------------------------------
    # 6. Création de vecteurs et tests d'addition, produit scalaire, multiplication par un scalaire
    # --------------------------------------------------
    
    # Vecteur ligne
    v1 = Vector([[1.0, 2.0, 3.0]])  # Vecteur ligne 1x3
    print("\nVecteur v1 (ligne) :")
    print(v1)

    # Vecteur colonne
    v2 = Vector([[1.0], [2.0], [3.0]])  # Vecteur colonne 3x1
    print("\nVecteur v2 (colonne) :")
    print(v2)

    # Test de l'addition de vecteurs
    print("\nAddition de v1 et v1 (v1 + v1) :")
    print(v1 + v1)

    # Test du produit scalaire (v2 . v2)
    print("\nProduit scalaire de v2 et v2 (v2 . v2) :")
    print(v2.dot(v2))

    # Test de la multiplication par un scalaire
    print("\nMultiplication de v1 par un scalaire 2.0 (v1 * 2.0) :")
    print(v1 * 2.0)

    # Test de la multiplication par un scalaire dans l'autre sens (2.0 * v1)
    print("\nMultiplication de 2.0 par v1 (2.0 * v1) :")
    print(2.0 * v1)

    # --------------------------------------------------
    # 7. Erreurs attendues
    # --------------------------------------------------
    
    # Test d'addition de matrices de dimensions différentes (m1 + m2)
    try:
        print("\nTest d'addition de matrices de dimensions différentes (m1 + m2) :")
        print(m1 + m2)
    except ValueError as e:
        print(f"Erreur : {e}")
    
    # Test de produit scalaire avec des vecteurs de dimensions différentes (v1 . v2)
    try:
        print("\nTest de produit scalaire avec des vecteurs de dimensions différentes (v1 . v2) :")
        print(v1.dot(v2))
    except ValueError as e:
        print(f"Erreur : {e}")
    
    # Test d'addition d'un vecteur et d'une matrice
    try:
        print("\nTest d'addition d'un vecteur et d'une matrice (v1 + m1) :")
        print(v1 + m1)
    except TypeError as e:
        print(f"Erreur : {e}")

    # Test de création de vecteur avec une mauvaise forme (vecteur 2D non valide)
    try:
        print("\nTest de création d'un vecteur non valide (2D) :")
        v3 = Vector([[1.0, 2.0], [3.0, 4.0]])  # Vecteur 2x2, pas valide
        print(v3)
    except ValueError as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
