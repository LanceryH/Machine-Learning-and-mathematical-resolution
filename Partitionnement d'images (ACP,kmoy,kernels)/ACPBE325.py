#%%
from PIL import Image 
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import copy



#%%
def Esperance(X):  # Calcul l'Esperance de X
    """
    Parameters
    ----------
    X : array([])
        Matrice X de taille (m,n)
    Returns
    -------
    Esperance de X
    """

    m = np.shape(X)[0]
    return(np.sum(X)/m)


def Variance(X):  # Calcul la Variance de X
    """
    Parameters
    ----------
    X : array([])
        Matrice X de taille (m,n)
    Returns
    -------
    Variance de X
    """

    m = np.shape(X)[0]
    return(np.sum(((X-Esperance(X))**2)/(m-1)))


def centre_red2(E) : 
    l,c=np.shape(E)
    ind_G=np.mean(E, axis=0)
 
    Mcentrered= E-np.kron(ind_G,np.ones((l,1)))
   
    for i in range(c) :
        if np.linalg.norm(Mcentrered[:,i])!=0 :
            Mcentrered[:,i]=(1/np.linalg.norm(Mcentrered[:,i]))*Mcentrered[:,i]
       
       
    return Mcentrered
#%%


def ACPimg(tabl,q=0):

    Xrc=centre_red2(tabl)
    C=Xrc.T@Xrc
    
    
    n,d=np.shape(tabl)
    U=np.linalg.eig(C)[1] #3 racines réelles simples, C est donc diagonalisable
    D=np.diag(np.linalg.eig(C)[0])
    
    lambdas=np.linalg.eigvals(C)
    
        

    for i in range(np.shape(lambdas)[0]):
        
        if lambdas[i] >= np.sum(lambdas)/d:
            q=q+1
    Uq=np.zeros((d,q))
    
    for i in range(q):
        Uq[:,i]=U[:,i]
        
    Xq=Xrc@Uq
    return(Xq)






















