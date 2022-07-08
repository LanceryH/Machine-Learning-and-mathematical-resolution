# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:08:50 2022

@author: hugol
"""

import copy as copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import tkinter as tk

# Cette fonction permet de créer une matrice de LAPLACE rapidement
def Matrix_Laplace(N=10,A=4,B=-1,C=-1):
    M=B*np.eye(N)+A*np.diag(np.ones(N-1),1)+C*np.diag(np.ones(N-1),-1)
    return M

# Fonction de résolution avec la méthode du gradient à pas constant
def GPC(A,b,x0,epsilon) : 
    x=copy.copy(x0)
    d=b-A@x
    compteur=0
    y=x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) >epsilon and compteur <1000 : 
         y=copy.copy(x)
         t=-(d.T @ (A@x-b))/(d.T @A @d)
         x=x+t*d
         beta=(d.T @ A @ (A@x-b)) / (d.T @ A @ d)
         d=-(A@x-b)+beta*d
         compteur+=1
    #print("GPC : La convergence à {} près est obtenue pour {} itérations.".format(epsilon,compteur))
    return x

# Fonction projeté de x
def proj(x) : 
    xproj=np.maximum(np.minimum(x,0.5*np.ones(np.shape(x))),-0.5*np.ones(np.shape(x)))
    return xproj

# Fonction de résolution avec la méthode du gradient projeté
def Gproj(A,b,rho,x0,epsilon) : 
    x=copy.copy(x0)
    r=b-A@x
    compteur=0
    y=x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) >epsilon and compteur <1000 : 
         y=copy.copy(x)
         x=proj(x+rho*r)
         r=b-A@x
         compteur+=1
    return x

#Fonction de la solution
def U(x,t,c,Un0,Vn0,Vn0_int):
    return 0.5*(Un0(x-c*t)+Un0(x+c*t))+(1/(2*c))*(Vn0_int(x+c*t)-Vn0_int(x-c*t))
#Fonction calculant le premier Un1
def Un1(x,dt,dx,M,c,Un0,Vn0):
    A=Matrix_Laplace(M,1,-2,1)
    return Un0(x)+dt*Vn0(x)-A@Un0(x)*(-(c**2)*(dt**2)/(dx**2))*(dt**2)/2*(dx**2)

#Fonction calculant les M itérations de Un2 avec contraintes
def Un2_contrainte(x,Un00,Un01,M,c,dx,dt,teta):
    A=Matrix_Laplace(M,1,-2,1)
    #gamma permet de rendre le code plus lisible
    gamma=-(c**2)*(dt**2)/(dx**2)
    #Pi trouvé dans la partie théorique
    P0=-np.eye(M)-(gamma*A*teta/2)
    P1=2*np.eye(M)-(gamma*(1-teta)*A)
    P2=-P0
    #Puis on retourne l'aproximation de la solution de Un2 en fonction de Un1 et Un0
    return Gproj(P2,P1@Un01+P0@Un00,0.02,np.ones((M)),10**-4)

#Fonction calculant les M itérations de Un2 sans contraintes
def Un2_no_contrainte(x,Un00,Un01,M,c,dx,dt,teta):
    A=Matrix_Laplace(M,1,-2,1)
    #gamma permet de rendre le code plus lisible
    gamma=-(c**2)*(dt**2)/(dx**2)
    #Pi trouvé dans la partie théorique
    P0=-np.eye(M)-(gamma*A*teta/2)
    P1=2*np.eye(M)-(gamma*(1-teta)*A)
    P2=-P0
    #Puis on retourne l'aproximation de la solution de Un2 en fonction de Un1 et Un0
    return GPC(P2,P1@Un01+P0@Un00,np.ones((M)),10**-3)

def simulation_corde(Un0,Vn0,Vn0_int,contrainte=0,T=10,L=1,M=200,N=300,c=1,teta=1,Path="C:/Users/hugol/Desktop"):
    dt=T/N
    dx=L/M
    x=np.linspace(0,L,M)
    #alpha=1
    #M=np.max(np.abs(np.linalg.eigvals(P2)))
    #cvmax=2*alpha/(M**2)
    #Initialisation des conditions initiales
    Un00=Un0(x)
    Un01=Un1(x,dt,dx,M,c,Un0,Vn0)
    Un00[0]=Un00[M-1]=0
    Un01[0]=Un01[M-1]=0
    #Un sera la matrice contenant tout les Un02 calculé a chaque pas
    Un=np.zeros((N+1,M))
    Un[0,:]=Un00
    Un[1,:]=Un01
    solN=np.zeros((N+1,M))
    #Initialisation de compteurs 
    compteur_dt=2*dt
    compteur_itération=2
    #On applique l'itération N fois suivant le choix de l'utilisateur avec ou sans contraintes
    for compteur_itération in tqdm(range (N)):
        if contrainte ==0:
            Un02=Un2_no_contrainte(x,Un00,Un01,M,c,dx,dt,teta)
        else:
            Un02=Un2_contrainte(x,Un00,Un01,M,c,dx,dt,teta) 
        Un00,Un01=copy.copy(Un01),copy.copy(Un02)
        Un02[0]=Un02[M-1]=0
        Un[compteur_itération,:]=Un02
        solN[compteur_itération,:]=U(x,compteur_dt,c,Un0,Vn0,Vn0_int)
        compteur_dt+=dt      
        compteur_itération+=1
    if contrainte ==0:
        #On compare notre solution avec celle réelle
        print("erreur totale: ",np.linalg.norm(Un-solN))
        print("erreur pour le début(1s): ",np.linalg.norm(Un[:30,:]-solN[:30,:]))
    #On enregistre la video de la corde (Un)    
    affichage_enregistre(contrainte,Un,solN,dt,M,N,Path)
    return Un,solN


def affichage_enregistre(contrainte,Un,solN,dt,M,N,Path):
    x = np.linspace(0, 1, M)
    # initialise la figure
    fig = plt.figure() 
    line1, = plt.plot([], [],'--r') 
    line2, = plt.plot([], [],'--b') 
    #Si l'option contraintes vaut 1 alors on trace les contraintes
    if contrainte==1:
        plt.plot(x,0.5*np.ones((M)),'--b')
        plt.plot(x,-0.5*np.ones((M)),'--b')
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    if contrainte==0:
        plt.legend(["Corde aprox","Corde réelle"])
    else:
        plt.legend(["Corde aprox","mur"])
    #FuncAnimation permet d'enregistrer nos plots
    def animate(i): 
        y1 = Un[i,:] 
        if contrainte==0:
            y2 = solN[i,:]
            line2.set_data(x, y2)
        line1.set_data(x, y1)
        return line1,line2,
    ani = animation.FuncAnimation(fig, animate, frames=N, blit=True, interval=90, repeat=False)
    ani.save('{}\Be321vid{}.gif'.format(Path,contrainte), writer='pillow', fps=60)
    plt.close()
    print("la vidéo a été enregistré ici ",Path)
    return


#%% Simulation corde sans contraintes
#Path doit etre le chemin d'accés où vous souhaitez enregistrer la vidéo !

#Fonctions définissant les conditions initiales
def Un0(x):
    return 0.5*np.sin(np.pi*x)
def Vn0(x):
    return -4*np.sin(np.pi*x)
def Vn0_int(x):
    return 4*np.cos(np.pi*x)/np.pi

simulation_corde(Un0,Vn0,Vn0_int,0,T=10,L=1,M=200,N=300,c=1,teta=1,Path="C:/Users/hugol/Desktop/BE321_LANCERY_MIMOUNI_MOREL")  

#%% Simulation corde avec contraintes
#Path doit etre le chemin d'accés où vous souhaitez enregistrer la vidéo !

#Fonctions définissant les conditions initiales
def Un0(x):
    return 0.5*np.sin(np.pi*x)
def Vn0(x):
    return -4*np.sin(np.pi*x)
def Vn0_int(x):
    return 4*np.cos(np.pi*x)/np.pi

simulation_corde(Un0,Vn0,Vn0_int,1,T=10,L=1,M=200,N=300,c=1,teta=1,Path="C:/Users/hugol/Desktop/BE321_LANCERY_MIMOUNI_MOREL")  

#%% Simulation corde avec d'autre conditions initiales

#Fonctions définissant les conditions initiales
def Un0(x):
    return x*0
def Vn0(x):
    return 4*np.sin(np.pi*x*2)
def Vn0_int(x):
    return -4*np.cos(np.pi*x*2)/(2*np.pi)

simulation_corde(Un0,Vn0,Vn0_int,0,T=10,L=1,M=200,N=300,c=1,teta=1,Path="C:/Users/hugol/Desktop/BE321_LANCERY_MIMOUNI_MOREL")  

