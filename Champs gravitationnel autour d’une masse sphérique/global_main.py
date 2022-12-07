#%%
"""
                BE Ma322 : Champs gravitationnel autour d’une masse sphérique
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
#%%
"""
            Partie 1 : Trajectoire d’un photon autour d’une masse sphérique
"""
#%% 


def fa(r):
    return (c**2)*(1+K/r)

def fb(r):
    return (1+(K/r))**(-1)

def Aprim(r):
    return (-(c**2)*K)/(r**2)
    
def Bprim(r):
    return (K/(r**2))/((1+K/r)**2)

#Fonction f pour l'algorithme de Runge Kutta 4
def f(Y):
    
    #Initialisation des valeurs pour le calcul 
    r=Y[1]
    A=fa(r)
    B=fb(r)
    Ap=Aprim(r)
    Bp=Bprim(r)
    
    Yp=np.zeros(len(Y))
    #On calcule 1 à 1 les 8 éléments du veceur Yp à partir des formules données en page 1 de l'énoncé
    for i in range(4):
        Yp[i]=Y[i+4]
    Yp[4]=(-(Ap/A))*Y[4]*Y[5]
    Yp[5]=(-(Ap/(2*B)))*(Y[4]**2)-((Bp/(2*B))*(Y[5]**2))+((r/B)*(Y[6]**2))+(((r*((np.sin(Y[2]))**2))/B)*(Y[7]**2))
    Yp[6]=((-2/r)*Y[6]*Y[5])+(np.sin(Y[2])*np.cos(Y[2])*Y[7]**2)
    Yp[7]=((-2/r)*Y[7]*Y[5])-(((2*np.cos(Y[2]))/(np.sin(Y[2])))*Y[6]*Y[7])
    return Yp.reshape(-1,1)
    
def Traj_photon(y0,h,itermax,f):
    """
    Parameters
    ----------
    y0 : Array de taille (8,1)
        Données initiales
    h : Réel positif différent de 0
        Pas de la méthode
    itermax : int
        Nombre d’itération maximal
    Returns
    -------
    P : Array de taille (8,N) avec N le nombre d'itérations de l'algo de Runge-Kutta 4
        Contenant N résultats des itérations de RK4     
    """
    
    #Création de la matrice P et initialisation avec y0 en première colonne
    P=np.zeros((8,itermax))
    P[:,0]=y0.ravel() # .ravel() permet de passer d'un yO de taille (8,1) à une taille (8,) et donc de le mettre en tant que colonne de P
    
    #On récpère r dans la première colonne pour le critère d'arret
    r  = P[1,0]
    
    #On applique l'algorithme de Runge-Kutta 4 pour définir successivement les colonnes de la matrice P
    j=1
    while j<itermax and r>rs :
        k1=h*f(P[:,j-1])
        k2=h*f(P[:,j-1].reshape(-1,1)+k1/2)
        k3=h*f(P[:,j-1].reshape(-1,1)+k2/2)
        k4=h*f(P[:,j-1].reshape(-1,1)+k3) 
        Pi=(P[:,j-1]).reshape(-1,1)+(1/6)*(k1 + 2*k2 + 2*k3 + k4)
        P[:,j]=Pi.ravel()
        r=P[1,j]
        j+=1
    print("Matrice P générée !")
    return P  


#%% 


def Visualisation(P,rs,lim):
    """
    Parameters
    ----------
    P : Array de taille (8,N) avec N le nombre d'itérations de l'algo de Runge-Kutta 4
        Contenant N résultats des itérations de RK4. Obtenue en sortie de la fonction Traj_photon
    rs : float
        Rayon de Schwarzschild.
    lim : unsigned int 
        Permet de fixer les tailles des axes.

    Returns
    -------
    Permet de visualiser en 3D la trajectoire d’un photon de paramètres initiaux données par y0.
    """
    ax = plt.figure().add_subplot(projection='3d')

    #Tracé de la sphère
    R = 2
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    xs = R * np.cos(u) * np.sin(v)
    ys = R * np.sin(u) * np.sin(v)
    zs = R * np.cos(v)
    ax.plot_surface(xs, ys, zs, color = 'royalblue', alpha = 0.9 )
    
    
    #Création d'une matrice contenant les coordonnées cartésiennes 
    C=np.zeros((3,np.shape(P)[1]))
    
    #Conversion des coordonnées de spérique vers cartésien
    for i in tqdm(range(0,np.shape(P)[1])) :
        r=P[1,i]
        theta=P[2,i]
        phi=P[3,i]
        C[0,i]=r*np.sin(theta)*np.cos(phi)
        C[1,i]=r*np.sin(theta)*np.sin(phi)
        C[2,i]=r*np.cos(theta)
        
    #Tracé de la trajectoire
    ax.plot(C[0,:i],C[1,:i],C[2,:i],c="r")    
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-lim,lim)    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title("Trajectoire d’un photon autour d’une masse sphérique")




#%% Application numérique

c=1
G=1
M=1
K=-2*G*M/(c**2)

y0=np.array([1, 10, np.pi/2, 0, 0, -1.5, 0, -0.1]).reshape(-1,1)
h=10**(-2)
itermax=10**3     
rs = 2*G*M/(c**2)
lim=10

P=Traj_photon(y0,h,itermax,f)

C = Visualisation(P, rs, lim)


#%%
"""
Question 5
"""
#On teste pour différentes valeurs de d phi en conservant les autres valeurs initiales identiques
#%%
dphi = -0.011
#%%
dphi = -100
#%%
dphi = 0.05
#%%
y0=np.array([1, 10, np.pi/2, 0, 0, -1.5, 0, dphi]).reshape(-1,1)
P=Traj_photon(y0,h,itermax,f)
C = Visualisation(P, rs, lim)
    
#%% 
"""
Question 6. Application
"""

#%% 6. (b) et (c)

c=63.24*10**5
G=4*(np.pi)**2
M=1
K=-2*G*M/(c**2)

y0=np.array([1, 0.1, np.pi/2, 0, 0, -1, 0, -0.75]).reshape(-1,1)
h=10**(-3)
itermax=10**3   

rs = 2*G*M/(c**2)
lim=100


P=Traj_photon(y0,h,itermax,f)

#%% 6. (d) Recherche de phin le plus proche de -pi/2

#Simple algorithme où l'on, garde en mémoire la valeur quand la différence avec la valeur exacte est plus faible que la précédente en mémoire

vexa=-np.pi/2
vapp1=P[3,0]
diff =np.abs(vexa-vapp1)
for i in range (np.shape(P)[1]):
    vappi=P[3,i]
    diffplus1 = np.abs(vexa-vappi)
    if diffplus1<diff:
        vmin=vappi
        cpt=i
        diff=diffplus1
        
print("\nPhi n le plus proche de -pi/2 est :",vmin,"\n\nLe rn correspondant est :",P[1,cpt], "atteint pour n =",cpt)
        
#%% 6. (d) v.

rinit=0.1
err=np.zeros((1,np.shape(P)[1]))
for i in range (np.shape(P)[1]):
    alpha1 = np.arctan(-P[1,i]/rinit)
    alpha2 = np.arctan(P[5,i]/(P[1,i]*P[7,i]))
    err[0,i]=alpha1-alpha2
    
moy=np.sum(err)/np.shape(P)[1]
print(moy)


#%%
"""
            Partie 2 : Expression numérique de la relation entre phi et r
"""

# %%
def f(x):
    return (np.pi/2)*np.sin(np.pi*x/2)

# %%
def Newton_Cotes4(f,a,b,n):
    S=0
    #On rentre les valeurs trouver de W
    W=[7/90,16/45,2/15,16/45,7/90]
    #On crée les N valeurs dans l'intervalle de l'intégrale
    x=np.linspace(a,b,n)
    for j in range (1,n):
        for i in range(len(W)):
            #On applique l'algorithme de NewtonCotes4
            S+=W[i]*f(x[j])*(x[j]-x[j-1])
    return S    

# %%
#On crée ici les différentes "précision" pour les comparer ensuite
n=np.linspace(50,1000,20)
a=0
b=1
Sol_exact=1
Ef=[]
#On enregistre alors chaque solution trouvé par l'algo comparé a la valeur réelle
for i in range (len(n)):
    Sol_appro=Newton_Cotes4(f,0,1,int(n[i]))
    Ef.append(Sol_appro-Sol_exact)
print(Ef)
plt.figure()
plt.title("Erreur relative tracer log.")
plt.loglog(n,Ef,'r')
plt.show()
# %%
def Newton_Cotes4(f,a,b,n,k):
    S=0
    W=[7/90,16/45,2/15,16/45,7/90]
    x=np.linspace(a,b,n)
    for j in range (1,n):
        for i in range(k+1):
            S+=W[i]*f(x[j])*(x[j]-x[j-1])
    return S   
 
#On définit Ar Br f(x) et Intphi
def A(r):
        return (c**2)*(1+K/r)
def B(r):
        return (1+K/r)**(-1)
def f(x):
        r0=0.0046547454
        if x==r0:
           return 0
        else:
            #application de l'equation de l'intégrale
            return(1/x)*(B(x)/(((A(r0)*x**2)/(A(x)*r0**2))-1))**0.5
    
def Int_phi(r,r0,N,k):
        return Newton_Cotes4(f,r0,r,N,k)       
            
#On défnit la distance max
max=50
#On définit les constantes
G=4*np.pi**2
c=63239.7263
M=1
K=-2*G*M/c**2
plt.figure()
plt.title("trajectoire d'un rayon de lumière \n passant à proximité du Soleil")
#On initialise la longeur d'étude de l'intégrale ou du probleme ici r et r0
r0=0.0046547454
n=r0
r=r0+n
while (max+1)*r0>r :
    #Tant qu'on atteint pas la limite d"étude on continue le calcul et la boucle
    #Tout en itérant a chaque fois le r
    Sol_appro=Int_phi(r,r0,1000,4)
    #On trace le résultat
    Photon=plt.scatter(r*np.cos(Sol_appro),r*np.sin(Sol_appro),color="red",marker=".")
    #On itère
    n+=r0
    r=r0+n
Soleil=plt.scatter (r0*np.cos(0),r0*np.sin(0),color="b",marker="o",label="Soleil")
plt.xlabel('x')
plt.ylabel('y')
plt.legend((Soleil,Photon),("Soleil", "Photon"),scatterpoints=1,loc='upper left',fontsize=10)
plt.show()
