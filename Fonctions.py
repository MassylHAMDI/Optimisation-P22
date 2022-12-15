#________________________________________ Declaration de fonctions____________________________________________

#Importation des bibliothéque:
import numpy as np



# Fonction qui retourne le résidu 
def Residu(i, param):
    L1, L2, x, y  = param
    a,b = i[0],i[1]
    return [L1*np.cos(a)+L2*np.cos(a+b)-x,L1*np.sin(a)+L2*np.sin(a+b)-y]


#____________________________________________________________________
# Retourne le carrée de la norme du résidu
def Residu_2(i, param):
    return np.linalg.norm(Residu(i,param))**2


#____________________________________________________________________
# Fonction qui retourne le gradient de la norme du residu R2
def dResidu_2(i,param):
    L1, L2, x, y  = param
    a,b = i[0],i[1]
    dx = 2 * L1 * x * np.sin(a) - 2 * L1 * y * np.cos(a) + 2 * L2 * x * np.sin(a + b) - 2 * L2 * y * np.cos(a + b)
    dy = -2 * L1 * L2 * np.sin(b) + 2 * L2 * x * np.sin(a + b) - 2 * L2 * y * np.cos(a + b)
    return dx, dy 


#____________________________________________________________________
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
    
#____________________________________________________________________
# Méthode de Gradient avec le mécanisme qui ajuste alpha:
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
    point_X = [xn]

    # Tableau qui contiendra le valeur de x
    point_Y = [yn]
    
    # initialisé alpha à une const
    alpha_security = alpha

    # un indicateur pour savoir si le programme à convergé 
    etat = True

    # tant que dX est plus grand que la précision = eps
    # n ne dépasse pas le num d'itérations max notre boucle tourne 
    while (dX > eps and n < nmax):
        
        # calcule du gradient de la norme du résidu R2 pour les valeurs xn et yn 
        gradx, grady = DJ([xn, yn],param)

        f_num = J([xn, yn],param)
        
        alpha = alpha_security

        # mécanisme qui ajuste si besoin la valeur de alpha
        if J([xn - (alpha * gradx), yn - (alpha * grady)],param) >= J([xn, yn],param):
            alpha = alpha * 0.05 

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
        etat = False
        print( f"Le programme ne converge pas")

    # on retourne la liste des x et y 
    # on retourne le nombre d'iteration réaliser 
    return [point_X, point_Y], n , etat


# Méthode de Gradient sans le mécanisme qui ajuste alpha:
def grad_fixe_sansmeca(J,DJ, x0, y0, alpha, eps, nmax, param):

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
    point_X = [xn]

    # Tableau qui contiendra le valeur de x
    point_Y = [yn]
    

    # un indicateur pour savoir si le programme à convergé 
    etat = True

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
        etat = False
        print( f"Le programme ne converge pas")

    # on retourne la liste des x et y 
    # on retourne le nombre d'iteration réaliser 
    return [point_X, point_Y], n , etat


#____________________________________________________________________
# Méthode de Newton
def Newton(J, DJ, HJ, x0, y0, eps, nmax, param):
    """
    Notations:
    J : le carrée de la norme du résidu
    Dj : le gradient de la norme du résidu
    Hj : La matrice hessienne du résidu
    Xn et Yn : point initial  l’itération n.
    dX : pas de déplacement pour une itération.
    n et nmax : compteur d’itérations et nombre maximal d’itérations autorisé. 
    X0 et Y0 : point de départ de l’algorithme.
    eps : : critère de précision.

    """

    # on initialise Xn à x0
    xn = x0

    # on initialise Yn à Y0
    yn = y0
    
    #dX initialisé à 1
    dX = 1    
    
    #n nombre d'iteration initialisé à 0
    n = 0
    
    #on initialise une liste avec la premiere valeur de xn 
    #qu'on augmentera au file de la boucle 
    Listex = [xn]
    Listey = [yn]
    
    # un indicateur pour savoir si le programme à convergé 
    etat = True
    
    #tant que dX est plus grand que la précision = eps et qu'on a
    #pas dépassé le num d'itérations
    while(dX > eps and n < nmax):
        
        # calcule du gradient de la norme du résidu R2 pour les valeurs xn et yn 
        gradx, grady = DJ([xn,yn],param)

        # On le transforme en numpy array
        grad = np.array([gradx,grady])
        
        # calcule de la matrice hessienne
        hessienne = HJ([xn,yn],param)
        inv_hessienne = np.linalg.inv(hessienne)
        
        # Calcule de delta  delta = -grad * inv(hess)
        delta = np.matmul(-1*grad,inv_hessienne)

        # calcule de xn et yn pour chaque iteration
        xn = xn + delta[0]   
        yn = yn + delta[1]

        # On ajoute xnet yn à la liste à chaque iteration
        Listex.append(xn)
        Listey.append(yn)
        
        # calcule de la norme de delta
        dX = np.linalg.norm(delta)
        
        #on incrémente n
        n = n+1

    # indicateur de convergence 
    # si notre programme ne converge pas on aura un message d'erreur dans la console principale 
    if (n >= nmax):
        etat = False
        print( f"Le programme ne converge pas")

    # on retourne la liste des x et y 
    # on retourne le nombre d'iteration réaliser   
    return [Listex,Listey],n,etat


#____________________________________________________________________
# Cette Fonction nous permet d'avoir une droite 
def Trajectoire(xydepart,xyfinal):
    x_start , y_start = xydepart
    x_final , y_final = xyfinal
    
    a = (y_final - y_start)/(x_final - x_start)
    b = y_start - a * x_start 
    x = np.linspace(x_start, x_final, 8)
    y = a*x - b
    
    return x,y


#____________________________________________________________________
# Cette fonction calcule la position x,y du bras articulé 
def position(theta1,theta2):
    x = []
    y = []

    L1 = 2
    L2 = 2
    # ici on suppose que len(theta1) est égal à que len(theta2)
    for i in range(len(theta1)):

        x_ = np.cumsum([0,
                    L1 * np.cos(theta1[i]),
                    L2 * np.cos(theta1[i]+theta2[i])])
                    
        y_ = np.cumsum([0,
                    L1 * np.sin(theta1[i]),
                    L2 * np.sin(theta1[i]+theta2)[i]])

        x.append(x_)
        y.append(y_)

    # Pour facilité la manipulation des tableaux on les mets en array
    x = np.array(x)
    y = np.array(y) 

    # retourne listes des position x et y   
    return x,y

