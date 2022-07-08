# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:35:53 2022

@author: hugol
"""
from PIL import Image 
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from ACPBE325 import *
import copy
from tqdm import tqdm
import random as rnd

def esperance(V) : #renvoie l'esperance d'un vecteur V    
    return np.mean(V)
def variance(V):   #renvoie la variance d'un vecteur V en passant par l'écart type (standart deviation) calculé par numpy
    return np.std(V)**2

def centre_red(R):    # renvoie une matrice centrée réduite a partir d'une matrice de base
                      # le terme Aij se calcule en soustrayant la moyenne de la colonne j à Aij
                      # puis en divisant par l'écart type de la colonne j
    m,n = np.shape(R)
    Rcr = np.zeros((m,n))
        
    for i in range(n):        
        xbar = esperance(R[:,i])        
        sigma = variance(R[:,i])**0.5
        
        for j in range(m):            
            Rcr[j,i] = (R[j,i] - xbar)/sigma
            
    return Rcr
def centre_red2(E) : 
    l,c=np.shape(E)
    ind_G=np.mean(E, axis=0)
    Mcentrered= E-np.kron(ind_G,np.ones((l,1)))
   
    for i in range(c):
        if np.linalg.norm(Mcentrered[:,i])!=0 :
            Mcentrered[:,i]=(1/np.linalg.norm(Mcentrered[:,i]))*Mcentrered[:,i]
    return Mcentrered

#%%
#Xrc=centre_red2(X)

def choixpts(image, nbpts) :

    L=[] #création d'un matrice qui contiendra tout les pixels choisi uniquement(480,nbpts,3)
    L_tot=np.zeros(np.shape(image)) #création d'un matrice de taille (480,640,3)  
    for i in range (1,np.shape(image)[0]-1):
        Pts_random=np.random.randint(1, np.shape(image)[1]-1, nbpts) #récupération d'un pixel random en évitant les bords
        L.insert(i,np.array(image[i,Pts_random,:])) #on insert dans la matrice vierge le pixels choisi
        L_tot[i,Pts_random]=image[i,Pts_random] #on remplace dans la matrice vierge le pixels choisi
    L=np.array(L) #L etait une liste on le repasse en array() pour utiliser cv2
    cv2.imwrite("L_tot.jpg",L_tot) #création de l'image de L_tot
    cv2.imwrite("L.jpg", L) #création de l'image de L

    return L,L_tot


def Kaiser(X):
    n,d=np.shape(X)
    C=Xrc.T@Xrc
    S=np.linalg.eigvals(C)
    lambdas=np.sum(S)
    q=0
    for i in range(np.shape(S)[0]):
        if S[i] >= (1/d)*lambdas:
            q=q+1
    return q

#q=Kaiser(X)

#%%
def data_pixels(img, L,nbpts) :
    print("Récupération des datas de l'image en cours ...")
    n,m,p=np.shape(img)
    k=0
    data_img=np.zeros((np.shape(L)[0]*nbpts,8)) #création de la matrice initiale de taille(np.shape(L)[0]*nbpts,8)
    for i in tqdm(range (n)):
        for j in range(m):#Suivant le pixel paroucur dans la matrice L_tot qui contient uniquement les pixels randoms de l'image a leur position
                if np.mean(L[i,j])!=0: #si le pixel est pas noir alors on le rentre dans le tableau de datas
                    data_img[k,0]=i
                    data_img[k,1]=j
                    data_img[k,2]=img[i,j,0]
                    data_img[k,3]=img[i,j,1]
                    data_img[k,4]=img[i,j,2]
                    data_img[k,5]=(int(img[i+1,j+1][0])+int(img[i,j+1][0])+int(img[i-1,j+1][0])+int(img[i+1,j][0])+int(img[i,j][0])+int(img[i-1,j][0])+int(img[i+1,j-1][0])+int(img[i,j-1][0])+int(img[i-1,j-1][0]))//9
                    data_img[k,6]=(int(img[i+1,j+1][1])+int(img[i,j+1][1])+int(img[i-1,j+1][1])+int(img[i+1,j][1])+int(img[i,j][1])+int(img[i-1,j][1])+int(img[i+1,j-1][1])+int(img[i,j-1][1])+int(img[i-1,j-1][1]))//9
                    data_img[k,7]=(int(img[i+1,j+1][2])+int(img[i,j+1][2])+int(img[i-1,j+1][2])+int(img[i+1,j][2])+int(img[i,j][2])+int(img[i-1,j][2])+int(img[i+1,j-1][2])+int(img[i,j-1][2])+int(img[i-1,j-1][2]))//9  
                    k=k+1            
    return data_img

#%%

def ACPimg(tabl,q=2):
    Xrc=centre_red2(tabl) #on centre réduit les données
    C=Xrc.T@Xrc
    n,d=np.shape(tabl) #on récupere la shape du tabl
    U=np.linalg.eig(C)[1] #3 racines réelles simples, C est donc diagonalisable
    Uq=np.zeros((d,q))
    for i in range(q):
        Uq[:,i]=U[:,i]      
    Xq=Xrc@Uq #on calcul Xq contenant les catégories du tabl centré réduit
    return(Xq)

def ACPond(X,q,W)  :
    """
    Parameters
    ----------
    X : array([]).
        Matrice X de taille (m,n).
    q : Nombre d Kaiser.
    W : array([]).
        Matrice diagonale de taille (m,n). Elle sert à donner plus d'importance
        à certains pixels/couleurs/positions lors de l'exécution de l'ACP. 
    Returns
    -------
    Xqp : array([]).
        Matrice Xqp de taille (m,q), ACP pondérée de X.
    """
    s=np.sum(W)
    Xrc=centre_red(X)
    Cw=(1/s)*(Xrc@W).T @ Xrc@W
    # Cw devient la nouvelle matrice de covariance de l'ACP, on effectue ensuite
    # la même démarche que l'ACP classique.
    U,S,Vt=np.linalg.svd(Cw)
    Uq=U[:,:q]
    Xqp=Xrc@Uq
    return Xqp

#%%

def muf(S1):
    """
    Parameters
    ----------
    S1 : 
        Partition des lignes de A (de taille m,n) en k catégorie. A est de la forme : [[], [], .., []]
    Returns
    -------
    muf : Matrice de taile (k,n)
        Chaque ligne de la matrice est le barycentre de tous les points d'un catégorie S[k]
    """
    return np.array([np.sum(s, axis=0, dtype=float) / len(s) for s in S1])

def Kmoy(A,k,epsilon,afficher=False):
    """
    Parameters
    ----------
    A : Matrice de taille (m,n) 
        A contien la base de donnée sur laquelle réaliser l'algorithme des k-moyens
    k : Int
        Le nombre de groupe / catégorie que l'on souhaite créer
    epsilon : Float
        condition d'arrete de l'algorithme
    afficher : optional
        if True affiche en 2D les données colorées selont leur partition. The default is False.
    Returns
    -------
    S : Matrice 
        La meilleur partition des lignes de A
    """
    m,n=np.shape(A) # On récupère les dimensions de la Matrice sur laquelle on réalise l'algorithme des k-moyens
    colour = ['b', 'g', 'r' , 'c' , 'm' , 'y', 'k', 'w', 'pink'] # on stocke des couleurs en vu d'un affichage
   
    # on initialise la liste contenant les barycentres 
    muo=[]
    
    # on choisie k ligne différente de A au hasard
    for i in range(k):
        r=rnd.randint(0,m)
        muo.append(A[r,:])
    compteur=0 #Permet de connaitre le nombre d'itérations de l'algo
    
    # Dans cette boucle, on effectue les étapes 2 à 5
    # A chaque itération, le delta entre les mu est mis à jour.
    
    delta=np.array(muo) #On initialise la valeur du delta entre mu(f-1) et mu(f). C'est notre variable "mémoire" de l'itération n-1
    # Tant que l'algorithme ne converge pas, c'est à dire que les barycentres se deplacent encore :
    # On ittère à nouveau l'algorithme
    while np.linalg.norm(delta) > epsilon:
    
        S1=[[]for i in range(k)] #On initialise notre partition des lignes de A en k catégorie. C'est S1 qui servira aux différents calculs
        S=[[]for i in range(k)] # S sera la même matrice que S1, elle contiendra en plus une colonne remplie du numéro de la partition associé 
        Sind=[[]for i in range(k)] # Sind sera la matrice contenant 
        
        # Pour chacune des lignes de A, on calcule sa distance aux différents barycentres
        # Puis on ajoutre cette ligne à la catégorie dont le barycentre est le plus proche
        for i in range(m):
            D=[np.linalg.norm(A[i,:]-muo[p]) for p in range(k)] # On calcule les distances entre les lignes de A et les k-barycentres avec la norme 2
            indice= D.index(min(D)) # On récupère l'indice du barycentre le plus proche
            Ai=np.append(A[i,:],[[indice]]) # On ajoute cette indice là à la fin de la ligne correspondante  
            S[indice].append(Ai) # On ajoute à la partition correspondante la ligne de A modifiée
            S1[indice].append(A[i,:]) #On ajoute à la partition correspondante la ligne de A sans modification
            Sind[indice].append(i)
            if afficher : # Affichage des nuages de points coloriés selon leur catégorie
                plt.scatter(A[i,:][0], A[i,:][1], c=colour[indice])
        
        # Il est possible qu'un partition de S1 soit vide, pour éviter les problèmes on ajoute un point 0 dans ces partitions vides.
        # Ce point n'aura pas d'impact sur les barycentres muo      
        while [] in S1:
            S1.remove([])
            S1.append([np.zeros(n)])
        
        # On calcule (delta) le déplacement moyen des barycentres pour la condition d'arrêt ainsi que les nouveaux barycentres (muo)    
        delta=np.array(muo)-muf(S1)
        muo=muf(S1)
        
        compteur+=1 # Mise à jour du compteur
    print ("L'algorithme des K-moyens converge pour k={}, au bout de : {} itération(s)".format(k,compteur))
    if afficher : # Affichage des titres et sous-titres
        plt.suptitle("Affichage 2D de l'algorithme des K-moyens")
        plt.title("pour k={}, on à {} itération(s)".format(k,compteur))
    return S  , Sind
 
    
 #%%

def Masque(S,image,nbpts,data_pix):    
    imagef=np.zeros(np.shape(image)) #création d'une matrice de meme taille que l'image
    c=data_pix
    for i in S[0]:
            imagef[int(c[i][0]),int(c[i][1])]=[50,50,200] #on impose la couleur du pixel de la catégorie trouvé par l'ACP
 
    for j in S[1]:
            imagef[int(c[j][0]),int(c[j][1])]=[50,200,50] #on impose la couleur du pixel de la catégorie trouvé par l'ACP      
 
    return imagef
    
  



  

#%%

def RemplissageMasque(imgmasqueponctuel):
    print("Création du masque ...")
    mask=cv2.cvtColor(imgmasqueponctuel,cv2.COLOR_BGR2GRAY)
    mask[mask>0]=255
    mask=255*np.ones(np.shape(mask))-mask
    imgmasque=cv2.inpaint(imgmasqueponctuel,np.uint8(mask),3,cv2.INPAINT_NS)
    print("terminé !\n")
    return imgmasque

def Masquetot(img,imgmasque):
    print("Création du masque global en cours ...")
    n,m,p=np.shape(img)
    imageselec1=np.ones((n,m,p))*255 #création de l'image du masque global1 en fond blanc
    imageselec2=np.ones((n,m,p))*255 #création de l'image du masque global2 en fond blanc
    for i in tqdm(range (n)):
        for j in range(m):
            if np.mean(imgmasque[i,j][1]) < np.mean(imgmasque[:,:][1]): #Si le pixel est inférieur a la moyenne 
            #alors on récupere le veritable pixel de l'image pour le mettre dans l'image masque blanc
                imageselec1[i,j]=img[i,j]
            else:   
                imageselec2[i,j]=img[i,j]
    return imageselec1,imageselec2


#%%



