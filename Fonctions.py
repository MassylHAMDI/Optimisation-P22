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


# Fonction qui retourne le gradient de la norme du residi R2
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
def grad(J,DJ, x0, y0, alpha, eps, nmax, param):
    # Xn initialisé à x0
    xn = x0

    # Yn initialisé a y0
    yn = y0

    # dX initialisé à 1 pour entrer dans la boucle
    dX = 1

    # n nombre d'iteration
    n = 0
    
    point_X = [xn]
    point_Y = [yn]
    point_Z = []
    
    # tant que dX est plus grand que la précision = eps et qu'on a
    # pas dépassé le num d'itérations
    while (dX > eps and n < nmax):
        
        gradx, grady = DJ([xn, yn],param)
        f_num = J([xn, yn],param)
        
        # le nouveau point = point itération précédente - alpha * gradient
        xn = xn - (alpha * gradx)
        yn = yn - (alpha * grady)
        
        f_res = J([xn, yn],param)
        
        point_X.append(xn)
       
        point_Y.append(yn)
        
        point_Z.append(f_res)

        dX = abs(f_res - f_num)

        # on incrémente n
        n = n + 1
    return point_X, point_Y, point_Z


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

