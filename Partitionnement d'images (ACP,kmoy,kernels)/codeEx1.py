

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kernels import *
 

P=np.array([[0,0,1],
            [1,0,1.5],
            [0,1,0.5],
            [1,1,1],
            [0.5,0.5,0]])

#faisons une liste des kernels importé
Liste_kernels=[kern_gauss,kern_sigmo,kern_ratio_quad,kern_mquad,kern_inv_mquad,kern_cauchy,kern_log]

def MCsurface(P):
    
    A=np.ones((np.shape(P))) #Création d'une matrice de 1 de shape (5,3)
    A[:,:2]=P[:,:2] #On construit A en récupérant les 2 premières colonnes
    Z=P[:,-1:] #On construit Z en récupérant la dernière colonne
    
    solMC=np.linalg.pinv(A)@Z #Calcul de la solution classique
    erreur=np.linalg.norm(A@solMC-Z) #Calcul de l'erreur
    print("erreur MCsurface classique: ", erreur)
    
    return solMC , erreur

def Visualisation(P,X,nb):
    
    n,m=np.shape(P) #Récupération de la shape de P
    discreteX=np.linspace(np.min(P[:,0])-0.5,np.max(P[:,0])+0.5,nb) #Descrétisation de x
    discreteY=np.linspace(np.min(P[:,1])-0.5,np.max(P[:,1])+0.5,nb) #Descrétisation de y
    discreteX,discreteY=np.meshgrid(discreteX,discreteY) #Passage en meshgrid pour le tracé
    
    fxy=(X[0]*discreteX+X[1]*discreteY+X[2]).T #Calcul de la surface solution
    
    #TRACE DE LA SURFACE ET DES POINTS
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')   
    surf = ax.plot_surface(discreteX,discreteY,fxy,label='Solution MC classique', color='blue', alpha=0.2)
    surf._facecolors2d=surf._facecolor3d #erreur de la fonction legend en 3d avec une surface
    surf._edgecolors2d=surf._edgecolor3d #cette solution vient de stackflow
    ax.legend()
    for i in range(n):
        ax.scatter(P[i][0],P[i][1],P[i][2],s=20) 
    plt.show()
    
    return()

Visualisation(P,MCsurface(P)[0],nb=5)


def MCKsurface(P,kern):
    
    n,m=np.shape(P) #Récupération de la shape de P
    A=np.ones((np.shape(P))) #Création d'une matrice de 1 de shape (5,3)
    A[:,:2]=P[:,:2] #On construit A en récupérant les 2 premières colonnes
    Z=P[:,-1:] #On construit Z en récupérant la dernière colonne
    
    K=np.zeros((n,n)) #initialisation de K 
    for i in range(n) : 
        for j in range(n) : 
            K[i,j]=kern(A[i,:].reshape((3,1)),A[j,:].reshape((3,1))) #calcul des Kij avec un kernel
    
    solMCK=np.linalg.pinv(K)@Z #calcul de la solution avec noyau
    erreur=np.linalg.norm(K@solMCK-Z) #Calcul de l'erreur

    return solMCK , erreur


def fK(c_etoile,discreteX,discreteY,A,n,kern): #Fonction calcul de la Matrice S (surface solution)
    s=0 
    for i in range(n) : 
        s+=c_etoile[i]*kern(A[i].reshape((3,1)),np.array([discreteX,discreteY,1],dtype=object).reshape((3,1)))
    return s


def VisualisationK(P,c_etoile,nb, kern):
    n,m=np.shape(P) #Récupération de la shape de P
    A=np.ones((np.shape(P))) #Création d'une matrice de 1 de shape (5,3)
    A[:,:2]=P[:,:2] #On construit A en récupérant les 2 premières colonnes
    discreteX=np.linspace(np.min(P[:,0])-0.5,np.max(P[:,0])+0.5,nb) #Descrétisation de x
    discreteY=np.linspace(np.min(P[:,1])-0.5,np.max(P[:,1])+0.5,nb) #Descrétisation de y
    discreteX,discreteY=np.meshgrid(discreteX,discreteY) #Passage en meshgrid pour le tracé
    
    fK(c_etoile,discreteX,discreteY,A,n,kern)
    
    S=np.zeros(np.shape(discreteX))
    for i in range(np.shape(discreteX)[0]) :
            S[i]=fK(c_etoile, discreteX[i],discreteY[i],A,n,kern)
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')   
    surf = ax.plot_surface(discreteX,discreteY,S,label='Solution avec noyau', color='blue', alpha=0.2)
    surf._facecolors2d=surf._facecolor3d #erreur de la fonction legend en 3d avec une surface
    surf._edgecolors2d=surf._edgecolor3d #cette solution vient de stackflow
    ax.legend()
    for i in range(n):
        ax.scatter(P[i][0],P[i][1],P[i][2],s=20)
    
    plt.show()
    
    return

#Calcul du kernel optimal
b=[] #List qui contiendra les différentes erreurs des kernels
for i in range (len(Liste_kernels)):
    b.append(MCKsurface(P,Liste_kernels[i])[1])

j=b.index(min(b)) #comparaison du meilleur kernel

VisualisationK(P,MCKsurface(P,Liste_kernels[j])[0],nb=20, kern=Liste_kernels[j])    
print("erreur optimale MCKsurface avec noyau {}: ".format(Liste_kernels[j]), MCKsurface(P,Liste_kernels[j])[1])

















