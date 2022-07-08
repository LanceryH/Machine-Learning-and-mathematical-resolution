# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm  # permet d'afficher les barres de chargement
import tkinter as tk
from PIL import Image, ImageTk

# %%  chargement des données

# données d'entrainement
train_data = np.loadtxt('mnist_train.csv', delimiter=',')

test_data = np.loadtxt('mnist_test.csv', delimiter=',')  # données de test


"""
________________________________________________________________________________________________________________________
____________________________________________/        \__________________________________________________________________
___________________________________________| Partie 1 |_________________________________________________________________
____________________________________________\________/__________________________________________________________________

"""

# %% Q3 Calcul des valeurs propres approchées de A.T*A en utilisant l'algorithme d'itération QR
# Utilisation du cas de l'apprentissage de la reconnaissance du chiffre 0

# %% Construction de A et y selonleur valeur déterminée aux questions 1 et 2

nombredetection = 0  # On fixe un chiffre manuscrit à detecter
# On selectionne la première colonne de la base de données d'entrainment qui correspond au chiffre
valeurs = train_data[:, 0]

# On crée des listes contenant les indices vrai ou faux.
# Si la première colonne de la ligne correspond au nombredetection cet indice son indice est enregistré dans la liste indceu
indiceu = np.where(valeurs == nombredetection)
# Sinon l'indice est enregistré dans la liste des indices faux indicef
indicev = np.where(valeurs != nombredetection)

u = train_data[:, 1:][indiceu]
v = train_data[:, 1:][indicev]

# On crée les matrices  A et y selon leurs expressions définies théoriquelment aux questions précédentes
A = np.block([[u, np.ones((len(indiceu[0]), 1))],
             [v, np.ones((len(indicev[0]), 1))]])

y = np.block([[np.ones((len(indiceu[0]), 1))],
             [-np.ones((len(indicev[0]), 1))]])


# %% Calcul des Valeurs propres

AtA = A.T @ A
n = 50  # nombre d'itération de l'algorithme de QR


def vpQR(A, n):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   A : Array                                                                 |      
    |       Matrice                                                               |      
    |   n : Int                                                                   |              
    |       Nombre d'itération de l'algorithme de calcul des valeurs propres,     |
    |       pour la méthode QR                                                    |  
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   vpqr : Array                                                              |
    |       Vecteur colonne contenant les valeurs propres pour la méthode QR      |
    |_____________________________________________________________________________|

    """
    Q, R = np.linalg.qr(AtA)
    for i in range(n):
        B = R @ Q
        Q, R = np.linalg.qr(B)
    vpqr = np.diag(B)
    return vpqr


# Calcul des valeurs propres. Vp contiendra les valeurs propre de A.T * A
vp = vpQR(AtA, n)


# %% Approximation du rang de A

def rangQR(vp):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   A : Array                                                                 |      
    |       Vecteur colonne contenant les valeurs propres pour la méthode QR      |                                                      
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   rang_qr : Int                                                             |      
    |       Rang d'une matrice dont on à obtenue les valeurs propres.             |                    
    |       Le rang est donné par le nombre de valeurs prores,                    |
    |       différentes non nulles                                                |
    |_____________________________________________________________________________|

    """
    vpl = list(vp)
    rang_qr = 0
    l1 = []
    for i in range(len(vpl)):
        if vpl[i] >= 10**(-5) and vpl[i] not in l1:
            l1.append(vpl[i])
            rang_qr += 1
    return rang_qr


print("\nLe rang de A en utilisant la méthode QR est :", rangQR(vp))


# %% Q4 Estimation du rang à partir du calcul des valeurs singulières

U, S, Vt = np.linalg.svd(AtA)
vsl = list(S)
rang_svd = 0
l2 = []

for i in range(len(vsl)):
    if vsl[i] > 10**(-10) and vsl[i] not in l2:
        l2.append(vsl[i])
        rang_svd += 1

print("\nLe rang de A en utilisant la méthode SVD est :", rang_svd)


# %% Q5 On considère une nouvelle matrice A que l'on nomme Aeps

epsilon = 10  # arbitraire
Aeps = AtA + epsilon * np.eye(785)


# %% a) Justifiions par le calcul Python que cette matrice Aeps est symétrique, définie, positive.

vp_eps = np.linalg.eigvals(Aeps)
AepsT = Aeps.T
vp_eps_l = list(vp_eps)

if (np.array_equal(Aeps, AepsT)) == True and 0 not in vp_eps_l:
    print("\nA(epsilon) est une matrice symétrique définie postive car elle est égale à sa transposée et ses valeurs propres sont non nulles.")


# %% b)

def factocholesky(A):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   A : Array                                                                 |      
    |       Matrice symetrique définie positive                                   |  
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   L : Array                                                                 |      
    |       Matrice triangulaire inférieur,                                       |
    |       résultant de la décompostion de Cholesky de A                         |
    |_____________________________________________________________________________|

    """

    n = np.shape(A)[0]
    # Création d'une matrice zero
    L = np.zeros(np.shape(A))
    L[0, 0] = np.sqrt(A[0, 0])
    # Réalise la décomposition de Cholesky
    for j in range(1, n):
        L[j, 0] = A[j, 0]/L[0, 0]
    for i in range(1, n):
        L[i, i] = np.sqrt(A[i, i]-np.sum(L[i, :i]**2))
        for j in range(i+1, n):
            L[j, i] = (A[j, i]-np.sum(L[j, :i]*L[i, :i]))/L[i, i]
    return L


def resoltrianginf(A, b):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   A : Array                                                                 |      
    |       Matrice triangulaire inférieure                                       |      
    |   b : Array                                                                 |  
    |       Vecteur colonne                                                       |          
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   x : Array                                                                 |      
    |       Vecteur colonne solution du système                                   |
    |_____________________________________________________________________________| 

    """

    l, c = np.shape(A)
    x = np.zeros(l)
    x[0] = b[0, 0]/A[0, 0]
    for i in range(1, l):
        x[i] = b[i, 0]
        for j in range(0, i):
            x[i] = x[i]-A[i, j]*x[j]
        x[i] = x[i]/A[i, i]
    return x


def resoltriangsup(A, b):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   A : Array                                                                 |      
    |       Matrice triangulaire supérieure                                       |      
    |   b : Array                                                                 |  
    |       Vecteur colonne                                                       |          
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   x : Array                                                                 |      
    |       Vecteur colonne solution du système                                   |
    |_____________________________________________________________________________|

    """

    l, c = np.shape(A)
    x = np.zeros(l)
    x[-1] = b[-1, ]/A[-1, -1]
    for i in range(l-2, -1, -1):
        x[i] = b[i, ]
        for k in range(i+1, l):
            x[i] = x[i] - A[i, k]*x[k]
        x[i] = x[i]/A[i, i]
    return x.reshape((l, 1))


def f(x, sol):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   A : Array                                                                 |      
    |       Matrice triangulaire supérieure                                       |      
    |   sol : Array                                                               |
    |        Vecteur solution de l'équation de A(epsilon)                         |          
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   T : Int                                                                   |      
    |       Valeur définissant la prédiction sur la valeur du chiffre manuscrit.  |                                
    |_____________________________________________________________________________|

    """

    w = sol[:-1, 0].reshape((784, 1))
    b = sol[-1, 0]
    Vhyp = w.T @ x + b
    return Vhyp


def resChol(nombredetection, epsilon):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   nombredetection : Int                                                     |  
    |       nombre à détecter                                                     | 
    |   epsilon : int>0                                                           |          
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   nombredetection : Int                                                     |          
    |       nombre à détecter                                                     |     
    |   soleps : Vecteur colonne                                                  |                  
    |       Contient                                                              |              
    |   txressite : Int                                                           |                      
    |       Le taux de réussite de détecion des chiffres par l'algorithme         |                     
    |   Mc : Matrice 2x2                                                          |      
    |       La matrice de confusion de la forme [[Nvp Nfn][Nfp Nvn]]              |      
    |       où Nvp le nombre de vrais positifs, Nfn le nombre de faux négatifs,   |      
    |       Nfp le nombre de faux positifs et Nvn le nombre de vrais négatifs     |      
    |_____________________________________________________________________________|

    """

# Question 5.b)i.

    # Création de u, v, A et y en fonction du nombre à détecter
    valeurs = train_data[:, 0]
    indiceu = np.where(valeurs == nombredetection)
    indicev = np.where(valeurs != nombredetection)
    u = train_data[:, 1:][indiceu]
    v = train_data[:, 1:][indicev]
    A = np.block([[u, np.ones((len(indiceu[0]), 1))],
                 [v, np.ones((len(indicev[0]), 1))]])
    y = np.block([[np.ones((len(indiceu[0]), 1))],
                 [-np.ones((len(indicev[0]), 1))]])

    # Calcul de sol(epsilon)
    Aeps = (A.T)@A + epsilon * np.eye(785)
    L = factocholesky(Aeps)
    b = A.T @ y

    # Résolution des systèmes triangulaires inférieur puis supérieur
    solinf = resoltrianginf(L, b)
    soleps = resoltriangsup(L.T, solinf)

# Question 5.b)ii.

    Nvp = 0  # nombre de vrais positifs
    Nfn = 0  # nombre de faux négatifs
    Nfp = 0  # nombre de faux positifs
    Nvn = 0  # nombre de vrais négatifs
    labels_test = test_data[:, 0]
    N = len(labels_test)

    for k in range(0, N):
        T = f(test_data[k, 1:], soleps)
        if T > 0 and labels_test[k] == nombredetection:
            Nvp = Nvp+1
        if T <= 0 and labels_test[k] == nombredetection:
            Nfn = Nfn+1
        if T > 0 and labels_test[k] != nombredetection:
            Nfp = Nfp+1
        if T <= 0 and labels_test[k] != nombredetection:
            Nvn = Nvn+1
    txreussite = (Nvp+Nvn)/N
    Mc = np.array([[Nvp, Nfn], [Nfp, Nvn]])

    return nombredetection, soleps, txreussite, Mc


nombredetection, soleps, txreussite, Mc = resChol(0, epsilon)

print('\nMatrice de confusion pour %s: ' % nombredetection + '\n', Mc)
print('Taux de réussite : ', txreussite)


# %% c) Calcul du taux de réussite ainsi que de la matrice de confusion associée pour chaque chiffre nombredetection compris dans l'intervale [0,9] pour epsilon=1

for i in tqdm(range(10)):
    nombredetection, soleps, txreussite, Mc = resChol(i, 1)
    print('\nMatrice de confusion pour %s : ' % nombredetection + '\n', Mc)
    print('Taux de réussite : ', txreussite, '\n')


# %% d) En faisant varier epsilon sur l’intervalle [e(-10), e(9)] et epsilon non nul on trace le taux de réussite en fonction de epsilon en utilisant resChol(0,epsilon).

# on découpe l'intervalle de epsilon en 50 parties
epsilon = list(np.linspace(0.5*10**-1, 10**4, 50))

taux_reussite = []
for i in tqdm(range(len(epsilon))):
    if epsilon[i] != 0:
        nmbdetection, seps, txreussite, Mconf = resChol(9, epsilon[i])
        taux_reussite.append(txreussite)


fig = plt.figure(1)
plt.plot(epsilon, taux_reussite, 'r')
plt.xlabel("Epsilon")
plt.ylabel('Taux de réussite')
plt.title("Taux de réussite en fonction du epsilon pour le chiffre 9")
plt.show()


# %% e) Pour chaque chiffre i appartenanta à [0,9] on cherche epsilon(i) qui maximise la détection du chiffre i

e0 = np.linspace(5.5*10**6, 9*10**6, 50)
e1 = np.linspace(1.9*10**8, 2.0*10**8, 50)
e2 = np.linspace(4.1*10**7, 4.3*10**7, 50)
e3 = np.linspace(10**-1, 10**5, 50)
e4 = np.linspace(1.7*10**7, 1.8*10**7, 50)
e5 = np.linspace(1.5*10**7, 1.75*10**7, 50)
e6 = np.linspace(1.88*10**8, 1.96*10**8, 50)
e7 = np.linspace(1.66*10**8, 1.8*10**8, 50)
e8 = np.linspace(0.5*10**-1, 10**3, 50)
e9 = np.linspace(0.5*10**-1, 10**4, 50)

epsilontot = np.stack((e0, e1, e2, e3, e4, e5, e6, e7, e8, e9), axis=1)
tauxreussite = []

for j in range(10):
    tauxreussite = []
    for i in tqdm(range(50)):
        if epsilontot[i, j] != 0:
            nmbdetection, seps, txreussite, Mconf = resChol(
                j, epsilontot[i, j])
            tauxreussite.append(txreussite)
    valeur_max = max(tauxreussite)

    print('\nTaux de réussite maximal pour le chiffre {} : {}'.format(j, valeur_max))

    epsopti = epsilontot[tauxreussite.index(valeur_max), j]
    print('Epsilon correspondant : ', epsopti, '\n')

    seps = resChol(j, epsopti)[1]
    np.save("soleps{}.npy".format(j), seps)
    print("Vecteur soleps{}.npy".format(j), "enregistré sur le pc")

# %% Q6 Rédaction du programme de détection des chiffres

# %% Création des fonctions de reconnaissance global
matrix_soleps = np.array([])
matrix_soleps = np.load('soleps0.npy')
for i in range(1, 10):
    matrix_soleps = np.block(
        [matrix_soleps, np.load('soleps{}.npy'.format(i))])


def fglobal(x, matrix_soleps):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   x : Array (784,1)                                                         |
    |       Image vectorisée d’un chiffre manuscrit                               |          
    |   matrix_soleps : Array (785,10)                                            |   
    |       Matrice contenant les vecteurs colonnes solepsi                       |                  
    |       (i appartennant à [0,9])                                              |          
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   TYPE : DESCRIPTION                                                        |                                
    |_____________________________________________________________________________|

    """
    return matrix_soleps[:-1, :].T @ x + matrix_soleps[-1, :].reshape((10, 1))


def reponse(x, matrix_soleps):
    """ 
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   x : Array (784,1)                                                         |
    |       Image vectorisée d’un chiffre manuscrit                               |          
    |   matrix_soleps : Array (785,10)                                            |   
    |       Matrice contenant les vecteurs colonnes solepsi                       |                  
    |       (i appartennant à [0,9])                                              |          
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   TYPE : Int                                                                |
    |       la fonction réponse donne la valeur prédite                           |                                
    |_____________________________________________________________________________|

    """
    rep = fglobal(x, matrix_soleps)
    s = np.max(rep)
    return int(np.where(rep == s)[0][0])


# %% Vérification global sur les données test (10 000 données)

l, c = np.shape(test_data)
cpt = 0
for i in tqdm(range(l)):
    label = test_data[i, 0]
    if reponse(test_data[i, 1:].reshape((784, 1)), matrix_soleps) == label:
        cpt += 1

print('\nLe taux de reconnaissance global est de {}'.format(cpt/l))


# %% Q7

plt.ioff()
fen = tk.Tk()
fen.title('Reconnaissance2chiffres')


def getEntry1_andshow():
    nligne_choisi = entree1.get()
    nligne_choisi = int(nligne_choisi)
    print(nligne_choisi)
    ligne_chiffre = test_data[nligne_choisi, 1:]
    matrix_ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
    image_PIL = ImageTk.PhotoImage(image=Image.fromarray(matrix_ligne_chiffre))

    label2.configure(image=image_PIL)
    label2.image = image_PIL
    label2.pack(padx=10, pady=10)

    textvar = tk.StringVar()
    textvar.set("L'algorithme voit un {}".format(
        reponse(ligne_chiffre.reshape((784, 1)), matrix_soleps)))
    label1.configure(textvariable=textvar)

    return()


Frame1 = tk.Frame(fen, borderwidth=2, relief="groove")
Frame1.pack(side="left", padx=10, pady=10)

Frame2 = tk.Frame(fen, borderwidth=2, relief="groove")
Frame2.pack(side="left", padx=10, pady=10)

Frame3 = tk.Frame(Frame1, borderwidth=2, relief="groove")
Frame3.pack(side="right", padx=5, pady=5)

Frame5 = tk.Frame(Frame1, bg="white", borderwidth=2, relief="groove")
Frame5.pack(side="top", padx=5, pady=5)

Frame6 = tk.Frame(Frame1, borderwidth=2, relief="groove")
Frame6.pack(side="bottom", padx=10, pady=10)

Frame4 = tk.Frame(Frame2, borderwidth=2, relief="groove")
Frame4.pack(side="top", padx=5, pady=5)


textvar = tk.StringVar()
textvar.set("L'algorithme voit un #")
label1 = tk.Label(Frame4, height=1,  font=("Calibri", 15),
                  textvariable=textvar, bg="white")
label1.pack()


value = tk.StringVar()
entree1 = tk.Entry(Frame1, textvariable="int", width=30)
entree1.pack(padx=10, pady=10)

btn1 = tk.Button(Frame5, height=1, width=20,
                 text="valider le n° ligne", command=getEntry1_andshow)
btn1.pack()


nligne_default = 1
ligne_chiffre = test_data[nligne_default, 1:]
ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
image_PIL = ImageTk.PhotoImage(image=Image.fromarray(ligne_chiffre))
label2 = tk.Label(Frame6, image=image_PIL)
label2.pack(padx=10, pady=10)

#n_ligne = input(int(print("choisir une ligne")))

fen.mainloop()


"""
________________________________________________________________________________________________________________________
____________________________________________/        \__________________________________________________________________
___________________________________________| Partie 2 |_________________________________________________________________
____________________________________________\________/__________________________________________________________________

"""
# %% Q1 Création d'une image moyenne des chiffres manuscrits 0


nbdetection = 0  # Nombre que l'on veut détecter. __fonction__
# Récuperation de la première colonne du fichier train_data.scv.
valeur = train_data[:, 0]
# Récuperation de l'indice de la ligne contenant le nbdetection.
indiceu = np.where(valeur == nbdetection)
# u est la matrice qui contient toutes les lignes contenant le nbdetection.
u = train_data[:, 1:][indiceu]

zeromoyen = []
l, c = np.shape(u)  # Récupération de la taille de la matrice u: (l,c).

for i in range(c):
    # Le vecteur zeromoyen récupère la moyenne de u.
    zeromoyen.append(np.mean(u[:, i]))

# Transformation en array() évite les erreurs pendant la manipulation de listes avec numpy
zeromoyen = np.array(zeromoyen)
# Redimensionnement de zero en une matrice image (28,28).
image = zeromoyen.reshape((28, 28))

cv2.imshow('chiffre', np.uint8(image))  # Affichage de l'image avec cv2
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


# %% Q2 Création d'une fonction chifremoy(train_data) que renvoie une image d'un chiffre moyens sélectionné

def chiffremoy(train_data, affichage=0):
    """
     ______________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   train_data : Fichier csv                                                  |  
    |       Conteient les données d'entrainement                                  |  
    |   affichage : 0/1                                                           |  
    |       (Active/Desactive le parametre d'affichage des images moyennes)       |  
    |_____________________________________________________________________________|
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   Umtot : Array                                                             |  
    |       Matrice contenant les 10 vecteurs des images des chiffres moyens      | 
    |_____________________________________________________________________________|

    """
    Umtot = []

    for i in range(10):  # Boucle qui fait varier le nombre de détection de 0 à 10 et applique le code de la question 1
        nbdetection = i
        valeur = train_data[:, 0]
        indiceu = np.where(valeur == nbdetection)
        u = train_data[:, 1:][indiceu]
        l, c = np.shape(u)

        for j in range(c):
            Umtot.append(np.mean(u[:, j]))
    Umtot = np.array(Umtot)
    Umtot = Umtot.reshape((10, c))

    if affichage == 1:  # Permet d'afficher les images des chiffres moyens si l'utilisateur le demande lors de l'utilisation de la fonction

        plt.ioff()
        fen = tk.Tk()
        fen.title('Affichage des chiffres moyens')
        # Création des 'box'  où seront affichées les images
        # Création de la 'box1'
        Framem = tk.Frame(fen, borderwidth=2, relief="groove")
        # Défintion de l'emplacement et de la taille
        Framem.pack(side="left", padx=5, pady=5)
        Frame0 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame0.pack(side="left", padx=5, pady=5)
        Frame1 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame1.pack(side="left", padx=5, pady=5)
        Frame2 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame2.pack(side="left", padx=5, pady=5)
        Frame3 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame3.pack(side="left", padx=5, pady=5)
        Frame4 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame4.pack(side="left", padx=5, pady=5)
        Frame5 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame5.pack(side="left", padx=5, pady=5)
        Frame6 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame6.pack(side="left", padx=5, pady=5)
        Frame7 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame7.pack(side="left", padx=5, pady=5)
        Frame8 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame8.pack(side="left", padx=5, pady=5)
        Frame9 = tk.Frame(Framem, borderwidth=2, relief="groove")
        Frame9.pack(side="left", padx=5, pady=5)

        # Boucle qui récupère, reshape et renvoie une image avec PIL des lignes de la matrice Umtot
        # La fonction locals permet de convertir un string en variable utilisable pour une fonction
        for i in range(10):
            nligne_default = i
            ligne_chiffre = Umtot[i]
            ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
            locals()["Image_PIL{}".format(i)] = ImageTk.PhotoImage(
                image=Image.fromarray(ligne_chiffre))

        # Boucle pour la création des 10 labels qui contiendront les images et qui seront entreposés dans chaque 'box'.
        for j in range(10):
            locals()["label{}".format(j)] = tk.Label(master=locals()[
                "Frame{}".format(j)], image=locals()["Image_PIL{}".format(j)])
            locals()["label{}".format(j)].pack()

        fen.mainloop()
    return Umtot


chiffremoy(train_data, 1)

# %% Q2 BONUS


Umtot = []
for i in range(10):  # Boucle qui fait varier le nombre de détection de 0 à 10 et applique le code de la question 1
    nbdetection = i
    valeur = train_data[:, 0]
    indiceu = np.where(valeur == nbdetection)
    u = train_data[:, 1:][indiceu]
    l, c = np.shape(u)

    for j in range(c):
        Umtot.append(np.mean(u[:, j]))
Umtot = np.array(Umtot)
Umtot = Umtot.reshape((10, c))

plt.ioff()
fen = tk.Tk()
fen.title('Affichage des chiffres moyens')
# Création des 'box'  où seront affichées les images
# Création de la 'box1'
Framem = tk.Frame(fen, borderwidth=2, relief="groove")
# Défintion de l'emplacement et de la taille
Framem.pack(side="left", padx=5, pady=5)
Frame0 = tk.Frame(Framem, borderwidth=2, relief="groove")
Frame0.pack(side="left", padx=5, pady=5)
Frame1 = tk.Frame(Framem, borderwidth=2, relief="groove")
Frame1.pack(side="left", padx=5, pady=5)
Frame2 = tk.Frame(Framem, borderwidth=2, relief="groove")
Frame2.pack(side="left", padx=5, pady=5)
Frame3 = tk.Frame(Framem, borderwidth=2, relief="groove")
Frame3.pack(side="left", padx=5, pady=5)
Frame4 = tk.Frame(Framem, borderwidth=2, relief="groove")
Frame4.pack(side="left", padx=5, pady=5)
# Boucle qui récupère, reshape et renvoie une image avec PIL des lignes de la matrice Umtot
# La fonction locals permet de convertir un string en variable utilisable pour une fonction
label2 = tk.Label(master=Frame2, text="Aprés optimisation =>")
label2.pack()

i, j = 6, 6
nligne_default = i
ligne_chiffre = Umtot[i]
ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
Image_PIL0 = ImageTk.PhotoImage(
    image=Image.fromarray(ligne_chiffre))
label0 = tk.Label(master=Frame0, image=Image_PIL0)
label0.pack()


i, j = 9, 9
nligne_default = i
ligne_chiffre = Umtot[i]
ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
Image_PIL1 = ImageTk.PhotoImage(
    image=Image.fromarray(ligne_chiffre))
label1 = tk.Label(master=Frame1, image=Image_PIL1)
label1.pack()


i, j = 6, 6
M_passage = np.zeros((28, 28))
nligne_default = i
ligne_chiffre = Umtot[i]
ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
ligne_chiffre = np.rot90(np.rot90(ligne_chiffre))
ligne_chiffre1 = ligne_chiffre.reshape((1, 784))
ligne_chiffre1 = (Umtot[9]+ligne_chiffre1)/2
ligne_chiffre1 = ligne_chiffre1.reshape((28, 28))
Image_PIL2 = ImageTk.PhotoImage(
    image=Image.fromarray(ligne_chiffre1))
label3 = tk.Label(master=Frame4, image=Image_PIL2)
label3.pack()


i, j = 9, 9
M_passage = np.zeros((28, 28))
nligne_default = i
ligne_chiffre = Umtot[i]
ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
ligne_chiffre = np.rot90(np.rot90(ligne_chiffre))
ligne_chiffre2 = ligne_chiffre.reshape((1, 784))
ligne_chiffre2 = (Umtot[6]+ligne_chiffre2)/2
ligne_chiffre2 = ligne_chiffre2.reshape((28, 28))
Image_PIL3 = ImageTk.PhotoImage(
    image=Image.fromarray(ligne_chiffre2))
label4 = tk.Label(master=Frame3, image=Image_PIL3)
label4.pack()


fen.mainloop()


def chiffremoy2(train_data, affichage=0):
    """
     _____________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   train_data : Fichier csv                                                  |  
    |       Conteient les données d'entrainement                                  |  
    |   affichage : 0/1                                                           |  
    |       (Active/Desactive le parametre d'affichage des images moyennes)       |  
    |_____________________________________________________________________________|
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   Umtot : Array                                                             |  
    |       Matrice contenant les 10 vecteurs des images des chiffres moyens      |
    |       avec les nouveaux 6 et le 9                                           |
    |_____________________________________________________________________________|

    """
    Umtot = []

    for i in range(10):  # Boucle qui fait varier le nombre de détection de 0 à 10 et applique le code de la question 1
        nbdetection = i
        valeur = train_data[:, 0]
        indiceu = np.where(valeur == nbdetection)
        u = train_data[:, 1:][indiceu]
        l, c = np.shape(u)

        for j in range(c):
            Umtot.append(np.mean(u[:, j]))
    Umtot = np.array(Umtot)
    Umtot = Umtot.reshape((10, c))

    # Permet d'afficher les images des chiffres moyens si l'utilisateur le demande lors de l'utilisation de la fonction

    i, j = 6, 6
    nligne_default = i
    ligne_chiffre = Umtot[i]
    ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))

    i, j = 9, 9
    nligne_default = i
    ligne_chiffre = Umtot[i]
    ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))

    i, j = 6, 6
    M_passage = np.zeros((28, 28))
    nligne_default = i
    ligne_chiffre = Umtot[i]
    ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
    ligne_chiffre = np.rot90(np.rot90(ligne_chiffre))
    ligne_chiffre1 = ligne_chiffre.reshape((1, 784))
    ligne_chiffre1 = (Umtot[9]+ligne_chiffre1)/2
    ligne_chiffre1 = ligne_chiffre1.reshape((28, 28))

    i, j = 9, 9
    M_passage = np.zeros((28, 28))
    nligne_default = i
    ligne_chiffre = Umtot[i]
    ligne_chiffre = np.array(ligne_chiffre).reshape((28, 28))
    ligne_chiffre = np.rot90(np.rot90(ligne_chiffre))
    ligne_chiffre2 = ligne_chiffre.reshape((1, 784))
    ligne_chiffre2 = (Umtot[6]+ligne_chiffre2)/2
    ligne_chiffre2 = ligne_chiffre2.reshape((28, 28))

    Umtot[6] = ligne_chiffre2.reshape((1, 784))
    Umtot[9] = ligne_chiffre1.reshape((1, 784))
    return Umtot


# %% Q3 Création d'une fontion Procuste(A,B)

def Procuste(A, B):
    """
     ______________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   A : Array (m,n)                                                           | 
    |       Matrice de même taille que B                                          | 
    |   B : TYPE (m,n)                                                            | 
    |       Matrice de même taille que A                                          | 
    |_____________________________________________________________________________| 
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   l : Int                                                                   |  
    |       Lambda, le rapport d'homothétie                                       |   
    |   X : TYPE                                                                  |  
    |       Une transformation orthogonale                                        |    
    |   t : Array                                                                 |  
    |       Vecteur de translation (vecteur colonne)                              |  
    |   erreur_transformation : Float                                             |  
    |       Erreur de transforamtion.                                             |  
    |       Donnée par la norme de Frobenius au carré de B-Phi(A)                 |  
    |       Avec Phi(A) = (lambda * X * A) + (t * u)                              |
    |_____________________________________________________________________________|

    """

    # Récupération de la dimension de la matrice B dans m,n.
    m, n = np.shape(B)

    aG = np.zeros((m, 1))  # Création vecteur nulle de taille m.
    for i in range(0, n):  # Boucle pour créer aG (Somme aj/n).
        # On transpose car  les données liées à un objet/individu sont sous forme de vecteurs ligne
        aG = aG+(1./n)*A[:, i][np.newaxis].T

    bG = np.zeros((m, 1))
    for i in range(n):  # Boucle pour créer bG (Somme bj/n).
        bG = bG+(1./n)*B[:, i][np.newaxis].T

    u = np.ones((1, n))  # vecteur unitaire de taille n.
    Ag = A-np.dot(aG, u)  # np.dot correspond au produit interne des vecteurs.
    Bg = B-np.dot(bG, u)
    P = np.dot(Ag, Bg.T)
    Ug, Sg, Vg = np.linalg.svd(P)  # Décomposition en valeurs singulières de P.

    # np.trace(##) Calcul la trace d'une matrice.
    l = np.trace(np.diag(Sg))/np.linalg.norm(Ag, 'fro')**2
    X = np.dot(Vg.T, Ug.T)
    t = bG-l*np.dot(X, aG)
    # np.linalg.norm(##,'fro') Calcul la norme de Frobenius.
    erreur_transformation = (np.linalg.norm(B-l*X@A+t@u, 'fro'))**2

    return l, X, t, erreur_transformation


# %% Q4 Implémentation d'une fonction de comparaison donnant en résultat la prediciton sur le chiffre manuscrit

def Comparaison(x, CM=chiffremoy(train_data)):
    """
     ______________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   x : Array                                                                 |  
    |       Un image de test_data sous la forme d'une ligne                       |  
    |   CM : Array                                                                |  
    |       Matrice contenant les 9 vecteurs des images des chiffres moyens       |  
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   veccomp : Array                                                           |  
    |       Matrice (10,1) contenant la norme de l'erreur de transformation,      |  
    |       entre le vecteur l'image de test_data et chacun des 10 vecteurs de CM |  
    |   resultat : Int                                                            |  
    |       Revoie la valeur d'un chiffre manuscrit,                              |  
    |       détectée par le programme pour une ligne donnée                       |  
    |       En comparant avec les vecteurs des images des chiffres moyens         | 
    |_____________________________________________________________________________|

    """

    veccomp = np.ones(10)  # Matrice unitaire de taille (10,10)

    for i2 in range(10):  # Boucle pour calculer l'erreur_transformation
        A = CM[i2, :].reshape((1, 784))  # Récupération de la ligne i2 dans CM
        B = x.T
        l, X, t, erreur_transformation = Procuste(
            A, B)  # Ref.Doc fonction procuste
        # Insertion des erreur_transformation dans veccomp
        veccomp[i2] = erreur_transformation

 # Code non demandé supplémentaire
    # Tentative de correction bonus de l'erreur 3/5
    # if resultat[0] == 3:
    #    if veccomp[6] < veccomp[7]:
    #        resultat[0] = 5

    # Récupere l'indice de l'emplacement dans veccomp de la valeur minimale
    resultat = np.where(veccomp == np.min(veccomp))[0]
    return veccomp, resultat


# Code non demandé supplémentaire

def tracedelacompa(i):
    """
     ______________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   i : Int                                                                   |  
    |       Indice de la ligne où l'on veut tracer l'erreur_transformation        |  
    |   affichage : 0/1                                                           |  
    |       (Active/Desactive) le parametre d'affichage des images moyennes       |  
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   graphe : plot                                                             |  
    |       Trace un nuage de points de l'erreur_transformation                   |  
    |       En fonction d'un chiffre                                              | 
    |_____________________________________________________________________________|

    """

    r, y = Comparaison(x=test_data[i, 1:] .reshape(
        (784, 1)), CM=chiffremoy(train_data))
    h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Test avec la ligne i et met en évidence le chiffre ayant la norme minimale. Doit correspondre au chiffre manuscrit
    plt.title("Test avec la ligne {} ".format(i))
    plt.plot(h, r, 'xr')
    plt.plot(y[0], r[y[0]], 'ob')
    plt.axhline(y=0, xmin=0, xmax=1, color='black')
    plt.xlabel("Chiffre à détecter")
    plt.ylabel("Erreur de transformation")
    plt.text(4.5, 10**5, "La ligne n°{} est détectée comme étant un {}\n".format(i,
             y[0]), horizontalalignment='center', color='b')
    plt.text(4.5, 0.5*10**5, "La véritable valeur de la ligne n°{} est {}".format(i,
             int(test_data[i, :] .reshape((785, 1))[0][0])), horizontalalignment='center', color='r')
    plt.show()
    return ()


itest = 489  # On prend une ligne i arbitrairement pour tester la fonction
print(tracedelacompa(itest))


# %% Q5 Rédaction du programme de détection des chiffres

def taux_reconnaissance():
    """
     ______________________________________________________________________________
    |   Explications                                                              |  
    |_____________________________________________________________________________|  
    |   La fonction utilise Comparaison() sur les 9999 valeurs                    |  
    |   du fichier test_data.scv et renvoie le taux_reconnaissance des chiffres   |  
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   taux_reconnaissance : plot                                                |  
    |       Trace un nuage de points de l'erreur_transformation                   |  
    |       En fonction d'un chiffre                                              |  
    |   barre de chargement : barre détat de la boucle dans le kernel             | 
    |_____________________________________________________________________________|    

    """

    fo = 0  # initialisation du compteur
    l, c = np.shape(test_data)  # l,c ligne et colonne de la matrice test_data
    # l = 100 #pour tester pour moins de valeur
    for i in tqdm(range(0, l)):  # Boucle avec une barre de chargement allant de 0 à l
        a, b = Comparaison(x=test_data[i, 1:] .reshape(
            (784, 1)), CM=chiffremoy(train_data))  # Ref.Doc fonction Comparaison
        # Condition si le nombre déduit par le programme est le bon
        if b[0] != int(test_data[i, :] .reshape((785, 1))[0][0]):
            fo += 1  # compteur d'erreur
    taux_erreur_pourc = fo*100/l  # Calcul du taux d'erreur avec une règle de trois

    print("\nle taux d'erreur global pour le fichier test est de {}% \n".format(
        taux_erreur_pourc))
    print("On a {} mauvaises détections pour {} valeurs".format(fo, l))

    taux_reco = 100-taux_erreur_pourc  # taux de réussite du programme

    return(taux_reco)


print(taux_reconnaissance())


"""
________________________________________________________________________________________________________________________
____________________________________________/        \__________________________________________________________________
___________________________________________| Partie 3 |_________________________________________________________________
____________________________________________\________/__________________________________________________________________

"""


# %% Q1 Redaction d'une fonction de reconnaissance des chiffres manuscrits utilisant les deux algorithmes précédents

# Initalisation des variables nécessaires
coeff = 10**-10
CM = chiffremoy(train_data)
# i=5
#x = train_data[i,1:].reshape((784,1))


def reconnaissance(x, coeff, SOLeps):
    """
     ______________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   x : Array                                                                 |  
    |       Vecteur colonne (784,1) contenant les données d'une ligne             |
    |        du fichier train_data                                                |  
    |   coeff : Float                                                             |  
    |       Réel flotant strictement positif                                      |
    |    SOLeps : Array                                                           |
    |       Matrice des solutions de la partie 1                                  |
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   graphe : Float                                                            |  
    |             Valeur attribué                                                 | 
    |_____________________________________________________________________________|

    """
    v = coeff * (1 / Comparaison(test_data[i, 1:].reshape((784, 1)), CM)[0])
    rep = fglobal(test_data[i, 1:].reshape(
        (784, 1)), SOLeps).reshape(np.shape(v))
    resultat = v + rep
    # print(resultat)
    return (resultat)


def reponsev2(x, SOLeps):
    """
     ______________________________________________________________________________
    |   => Parameters                                                             |  
    |_____________________________________________________________________________|  
    |   x : Array                                                                 |  
    |       Ligne de données du fichier train_data                                |  
    |_____________________________________________________________________________|  
    |   <= Returns                                                                |  
    |_____________________________________________________________________________|  
    |   ... : Int                                                                 |  
    |       Trace un nuage de points de l'erreur_transformation                   |  
    |       En fonction d'un chiffre                                              | 
    |_____________________________________________________________________________|

    """
    rep = reconnaissance(x, coeff, matrix_soleps)
    s = np.max(rep)
    return int(np.where(rep == s)[0][0])


# %% Q2 Calcule du taux de réussite de cet algorithme en fonction du coeff choisi

cpt = 0  # Initialisation du compteur
coeff = 6000  # Choix du coeeficient "optimisé"
l, c = np.shape(test_data)
for i in tqdm(range(l)):
    # Label est le chiffre manuscrit correspondant à la ligne de la base de donnée test_data
    label = test_data[i, 0]
    if reponsev2(test_data[i, 1:].reshape((784, 1)), matrix_soleps) == label:
        cpt += 1

print('\nLe taux de reconnaissance globale est de {}'.format(cpt/l))
print('Soit {}%'.format((cpt*100/l)))


# %%  Recherche de la valeur d'optimisation du taux de réussite (Non concluant)
coeff2 = np.linspace(10**-10, 20000, 10)

#l,c = np.shape (test_data)
l = 100
tauxreussite2 = []
for i in range(10):
    print(i+1, '/ 10')
    cpt = 0
    coeff = coeff2[i]
    for j in tqdm(range(l)):
        # Label est le chiffre manuscrit correspondant à la ligne de la base de donnée test_data
        label = test_data[j, 0]
        if reponsev2(test_data[j, 1:].reshape((784, 1)), matrix_soleps) == label:
            cpt += 1
        txreu2 = ((cpt/l))

    tauxreussite2.append(txreu2)
    valeur_max = min(tauxreussite2)

print('\nTaux de reconnaissance de', 1-valeur_max,
      "% \nCe taux est atteint en indice", tauxreussite2.index(valeur_max))
print("Soit pour un coeff de :", coeff2[tauxreussite2.index(valeur_max)])
