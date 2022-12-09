#________________________________________ Declaration de fonctions____________________________________________

#Importation des bibliothéque:
import numpy as np



# Fonction qui retourne le résidu 
def R(i, param):
    L1, L2, x, y  = param
    a,b = i[0],i[1]
    return [L1*np.cos(a)+L2*np.cos(a+b)-x,L1*np.sin(a)+L2*np.sin(a+b)-y]


# Retourne le carrée de la norme du résidu
def R2(i, param):
    return np.linalg.norm(R(i,param))**2


# Fonction qui retourne le gradient de la norme du residu R2
def dR2(i,param):
    L1, L2, x, y  = param
    a,b = i[0],i[1]
    dx = 2 * L1 * x * np.sin(a) - 2 * L1 * y * np.cos(a) + 2 * L2 * x * np.sin(a + b) - 2 * L2 * y * np.cos(a + b)
    dy = -2 * L1 * L2 * np.sin(b) + 2 * L2 * x * np.sin(a + b) - 2 * L2 * y * np.cos(a + b)
    return dx, dy 


# calcule de la matrice hessienne 
def H(i,param):
    L1, L2, x, y  = param
    a, b = i[0], i[1]
    dxx = 2 * (L1 * x * np.cos(a) + L1 * y * np.sin(a) + L2 * x * np.cos(a + b) + L2 * y * np.sin(a + b))
    dyy = 2 * L2 * (-L1 * np.cos(b) + x * np.cos(a + b) + y * np.sin(a + b))
    dxy = 2 * L2 * x * np.cos(a + b) + 2 * L2 * y * np.sin(a + b)
    dyx = 2 * L2 * x * np.cos(a + b) + 2 * L2 * y * np.sin(a + b)
    Hess = np.array([[dxx, dxy],
                     [dyx, dyy]])
    return Hess 
    

# Méthode de Gradient:
def grad_fixe(J,DJ, x0, y0, alpha, eps, nmax, param):

    """
    Notations:
    J : le carrée de la norme du résidu
    Dj : le gradient de la norme du résidu
    Xn et Yn : point initial  l’itération n.
    dX : pas de déplacement pour une itération.
    n et nmax : compteur d’itérations et nombre maximal d’itérations autorisé. 
    X0 et Y0 : point de départ de l’algorithme.
    alpha :  pas de recherche.
    eps : : critère de précision.

    """
    # on initialise Xn à x0
    xn = x0

    # on initialise Yn à Y0
    yn = y0

    # dX initialisé à 1 pour entrer dans la boucle
    dX = 1

    # n nombre d'iteration
    n = 0
    # Tableau qui contiendra le valeur de x
    point_X = []

    # Tableau qui contiendra le valeur de x
    point_Y = []
    
    
    # tant que dX est plus grand que la précision = eps
    # n ne dépasse pas le num d'itérations max notre boucle tourne 
    while (dX > eps and n < nmax):
        
        # calcule du gradient de la norme du résidu R2 pour les valeurs xn et yn 
        gradx, grady = DJ([xn, yn],param)

        f_num = J([xn, yn],param)
        
        # Calcule des nouveaux points 
        xn = xn - (alpha * gradx)
        yn = yn - (alpha * grady)
        
        f_res = J([xn, yn],param)
        
        # On ajoute xn à la liste à chaque iteration
        point_X.append(xn)

        # On ajoute xn à la liste à chaque iteration
        point_Y.append(yn)
        
        # Calcule de la nouvelle valeur de dX
        dX = abs(f_res - f_num)

        # on incrémente n
        n = n + 1
    
    # indicateur de convergence 
    # si notre programme ne converge pas on aura un message d'erreur dans la console principale 
    if (n >= nmax):
        print( f"Le programme ne converge pas")

    # on retourne la liste des x et y 
    # on retourne le nombre d'iteration réaliser 
    return [point_X, point_Y], n 


# Méthode de Newton
def Newton(J, DJ, HJ, x0, y0, err, n_max, param):
    #Xn initialisé à x0
    xn=x0
    
    #Yn initialisé a y0
    yn=y0
    
    #dX initialisé à 1
    dX = 1    
    
    #n nombre d'iteration initialisé à 0
    n = 0
    
    #on initialise une liste avec la premiere valeur de xn 
    #qu'on augmentera au file de la boucle 
    Liste1 = [x0]
    Liste2 = [y0]
    
    Liste3 = [J([xn, yn],param)]
    
    #tant que dX est plus grand que la précision = eps et qu'on a
    #pas dépassé le num d'itérations
    while(dX > err and n < n_max):
        
        gradx, grady = DJ([xn,yn],param)
        grad = np.array([gradx,grady])
        
        hess = HJ([xn,yn],param)
        inv_hess = np.linalg.inv(hess)
        
        delta = np.matmul(-1*grad,inv_hess)
        xn = xn + delta[0]   
        yn = yn + delta[1]
        
        Liste1.append(xn)
        Liste2.append(yn)
        Liste3.append(J([xn, yn],param))
        
        #dX = la norme de delta
        dX = np.linalg.norm(delta)
        
        #on incrémente n
        n = n+1
        
        
        list = [Liste1,Liste2,Liste3]
        
    return list

