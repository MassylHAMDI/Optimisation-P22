# Optimisation-P22

Ce projet implémente diverses méthodes numériques pour résoudre des problèmes d'optimisation, en se concentrant sur la cinématique inverse d'un bras robotique à 2 segments. Il comprend des fonctions pour calculer les résidus, les gradients et les matrices hessiennes, ainsi que des implémentations de la méthode de descente de gradient et de la méthode de Newton pour l'optimisation.

## Structure du Répertoire

- `Fonctions.py` : Contient les fonctions principales pour l'optimisation et les calculs du bras robotique.
- `Projet_Optimisation_Hamdi_Wang_Mokhbi_Gr...` : (Nom complet tronqué) Contient probablement le notebook ou le script principal du projet.

## Fonctionnalités

- Calcul du résidu pour un bras robotique à 2 segments
- Calcul du gradient et de la matrice hessienne
- Méthode de descente de gradient (avec et sans pas adaptatif)
- Méthode de Newton pour l'optimisation
- Génération de trajectoire pour le bras robotique
- Calcul de la position du bras robotique

## Fonctions Principales (dans Fonctions.py)

1. `Residu(i, param)` : Calcule le résidu pour des angles donnés et une position cible
2. `Residu_2(i, param)` : Calcule la norme au carré du résidu
3. `dResidu_2(i, param)` : Calcule le gradient du résidu au carré
4. `H(i, param)` : Calcule la matrice hessienne
5. `grad_fixe(J, DJ, x0, y0, alpha, eps, nmax, param)` : Implémente la descente de gradient avec pas adaptatif
6. `grad_fixe_sansmeca(J, DJ, x0, y0, alpha, eps, nmax, param)` : Implémente la descente de gradient sans pas adaptatif
7. `Newton(J, DJ, HJ, x0, y0, eps, nmax, param)` : Implémente la méthode de Newton pour l'optimisation
8. `Trajectoire(xydepart, xyfinal)` : Génère une trajectoire linéaire entre deux points
9. `position(theta1, theta2)` : Calcule la position du bras robotique étant donné les angles des articulations

## Utilisation

Pour utiliser ce projet, clonez le répertoire et exécutez le notebook Jupyter ou le fichier de script principal. Assurez-vous d'avoir installé les dépendances requises.
