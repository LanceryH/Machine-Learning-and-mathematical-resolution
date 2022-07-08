# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import random
import pandas as pd

Ebrut = np.genfromtxt("iris.csv", dtype=str, delimiter=",")  # données brutes
labelscolonne = Ebrut[0, : -1]
labelsligne = Ebrut[1:, -1]
E = Ebrut[1:, :-1] . astype("float")


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


def centre_red(R):  # Calcul de la matrice centrée réduite de R
    """
    Parameters
    ----------
    R : array([])
        Matrice X de taille (m,n)
    Returns
    -------
    Rcr : TYPE
        Matrice centrée réduite
    """

    m, n = np.shape(R)  # shape de la matrice d'entrée
    Rcr = np.zeros((m, n))  # matrice de zéros de taille (m,n)
    for i in range(n):
        Xa = R[:, i]  # Récupère la colonne de R
        # calcul de chaque colonne de la futur matrice centrée réduite
        Rcr[:, i] = (Xa-Esperance(Xa)) / np.sqrt(Variance(Xa))
    return (Rcr)


def approx(R, k):  # Calcul l'Esperance
    """
    Parameters
    ----------
    R : array([])
        Matrice X de taille (m,n)
    k : int
        k un entier non nul.
    Returns
    -------
    projk : Matrice des variance de R
    """

    X = centre_red(R)  # On centre réduit notre matrice d'entrée
    m, n = np.shape(X)  # on stock la forme m,n de la matrice centrée reduite
    # on applique une décomposition en éléments simple de X
    u, s, vt = np.linalg.svd(X)
    v = vt.T  # transpose vt
    projk = []
    for i in range(k):
        vk = v[:, i].reshape(len(v), 1)  # on récupère vk la colonne i de v
        Yk = X@vk  # on calcul Yk avec X et vk
        uj = u[:, i].reshape(len(u), 1)  # on récupère uj la colonne i de u
        # on insert dans projk la variance de Yk*uj
        projk.insert(i, Variance(Yk)*uj)
    # on utilise np.block pour construire notre matrice finale projk
    projk = np.block([projk[i] for i in range(k)])
    return(projk)


def Covariance(X, Y):  # Calcul de la Covariance de X avec Y
    """
    Parameters
    ----------
    X : array([])
        Matrice X de taille (m,n).
    Y : array([])
        Matrice Y de taille (m,n).
    Returns
    -------
    Covariance X
    """

    m = np.shape(X)[0]
    return(np.sum((X-Esperance(X))*(Y-Esperance(Y)))*(1/m))


def cor(X, Y):  # Fonction de Corélation entre deux matrices de même taille
    """
    Parameters
    ----------
    X : array([])
        Matrice X de taille (_,n).
    Y : TYPE
        Matrice Y de taille (_,n).
    Returns
    -------
    Cori : float

    """

    Cori = []
    m, n = np.shape(X)  # on stock la forme m,n de la matrice centrée reduite X
    for i in range(n):
        # Calcul de la corélation entre Y1 et Xi avec i variant de 0 à n-1
        Corik = Covariance(X[:, i], Y) / (Variance(X[:, i])*Variance(Y)) ** 0.5
        Cori.insert(i, Corik)  # on stock le résultat
    # la fonction est déjà en transposée
    return(Cori)


def correlationdirprinc(R, k):  # Ensemble des corélation de Yk et Xi
    """
    Parameters
    ----------
    R : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    Returns
    -------
    Cori : Matrice
        Renvoie la matrice des corélation de Yk et Xi
    """

    X = centre_red(R)
    Y = approx(X, k)
    Cori = []
    for i in range(k):
        # on applique la fonction Cori sur X et la colonne i de Y
        Corik = cor(X, Y[:, i])
        Cori.insert(i, Corik)  # On stock le résultat final
    return(Cori)


def ACP2D(R, labelsligne, labelscolonne):
    """
    Parameters
    ----------
    R : TYPE
        DESCRIPTION.
    labelsligne : TYPE
        DESCRIPTION.
    labelscolonne : TYPE
        DESCRIPTION.
    Returns
    -------
    Renvoie deux figures en 2D:
            Fig 1 – Valeurs des variances et pourcentages explicatifs de celles-ci pour les composantes
            Fig 2 – Représentation des données dans le plan (Y1, Y2) et le cercle de corrélation dans ce même plan (2D)
    """

    # __________Figure 1 : Valeurs des variances et pourcentages explicatifs de celles-ci pour les composantes principales_______________

    # Mise en page de la fenetre matplotlib

    # séparation  de la fenetre en deux pour afficher deux graphiques
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle('Figure 1 – Valeurs des variances et pourcentages explicatifs de celles-ci pour les composantes principales',
                 fontsize=12)  # Titre général de la fenêtre
    # Tracé de la variance des composantes principales

    X = centre_red(R)  # La matrice  des données est centrée et réduite
    # Décomposition SVD de la matrice X centrée réduite
    u, s, vt = np.linalg.svd(X)
    v = vt.T  # Transposition du vecteur vt provenant de la décompositon SVD. v Vecteur

    # On récupère les dimensions de X la matrice centrée réduite sur laquelle on travaille (Même dimension que R)
    m, k = np.shape(X)
    # On crée une matrice Yk de 2 colonnes. Elle nous servira à afficher les variances des composantes
    Yk = np.zeros(shape=(k, 2), dtype=object)
    for i in range(k):  # On remplit cette matrice Yk
        vk = v[:, i].reshape(len(v), 1)
        # La première colonne contient le label Yk
        Yk[i, 0] = '$Y_{}$'.format(i+1)
        # La seconde colonne contient la valeur de la variance correspondante
        Yk[i, 1] = Variance(X@vk)
    Yksum = 0
    for i in range(len(Yk)):
        Yksum += Yk[i, 1]
    ind = Yk[:, 0]  # Liste contenant les labels de la colonne 0 de la matrice Yk créée précédemment. Elle servira pour les labels des différentes variances
    # Création d'une nouvelle liste qui ne contient que les valeurs des différentes variances pour tracer le graphique en barres
    Ykvalues = list([])
    for i in range(k):  # On récupère les valeurs de variance contenue dans Yk et on les reconvertit du format object vers float
        Ykvali = float(Yk[i, 1])
        Ykvalues.append(Ykvali)
    labelsYkvalues = sorted(Ykvalues, reverse=True)

    # Création du graphique en barres
    ax1.bar(ind, Ykvalues, width=0.8, color='royalblue',)
    ax1.set_xlabel('Composantes', fontsize=9)
    ax1.set_ylabel('$Var(Y_{k})$', fontsize=9)
    ax1.set_title('Variances des composantes principales')

    # On place les valeurs de chacune des barres au dessus de celles-ci
    for index, value in enumerate(labelsYkvalues):
        ax1.text(index-0.2, value+0.04, str(round(value, 6)))

    # Création du graphique représentant le pourcentage de l’explication de la variance de chaque k−composantes principales

    # Calcul du pourcentage
    Ykpercent = []
    for i in range(len(Ykvalues)):
        Ykpercent.append((Ykvalues[i]*100)/Yksum)

    ax2.pie(Ykpercent, labels=ind, autopct=lambda Ykpercent: str(round(
        Ykpercent, 1)) + '%', shadow=True)  # "autopct" permet d'ajouter sur chacune des parts
    # la proportion en % qu'elle représente
    ax2.set_title('Participation à la variance totale')

    # _______________Figure 2 Représentation des données dans le plan (Y1, Y2) et le cercle de corrélation dans ce même plan_______________

    # Mise en page de la fenetre matplotlib
    fig2, (ax3, ax4) = plt.subplots(1, 2)
    plt.suptitle(
        'Figure 2 – Représentation des données dans le plan (Y1, Y2) et le cercle de corrélations dans ce même plan', fontsize=12)
    # Création du graphe en 2D qui représente la matrice de sortie de approx(R,2)
    ax3.set_title('Analyse en composantes principales pour k=2')
    ax3.set_xlabel('$Y_{1}$')
    ax3.set_ylabel('$Y_{2}$')

    # Utilisation de la méthode découverte en Ma 313 pour colorier chacun des points du nuage selon la valeur de leur labelsligne
    # Création  d'un  dictionnaire contenant chacun des labels ligne et assignation d'une couleur aléatoire.
    Dico = {}
    n = len(set(labelsligne))

    lbl = labelsligne
    App = (approx(R, 2))

    for i in labelsligne:
        if i not in Dico:
            color = "%06x" % random.randint(0, 0xFFFFFF)
            Dico[i] = color

    m, n = np.shape(R)
    # Tracé des points et de leur label
    for i in range(m):
        ax3.scatter(App[i, 0], App[i, 1], s=10*2**1, color='#' + Dico[lbl[i]])
    for i, txt in enumerate(lbl):
        ax3.annotate(txt, (App[i, 0], App[i, 1]))

    # Création du graphe de la matrice de sortie de la fonction correlationdirprinc(R,2)
    ax4.grid()
    ax4.set_title('Cercle de corrélation')
    ax4.set_xlabel('$Y_{1}$')
    ax4.set_ylabel('$Y_{2}$')
    ax4.set_xlim([-1.2, 1.2])
    ax4.set_ylim([-1.2, 1.2])
    draw_circle = plt.Circle((0, 0), 1, fill=False,
                             ls='--', lw='1.5', color='royalblue')
    ax4.set_aspect(1)
    ax4.add_artist(draw_circle)

    # Utilisation de la fonction correlationdirprinc pour récupérer les coordonnées de chacune des flèches à tracer dans le cercle de correlation
    Corx = correlationdirprinc(R, 2)[0]
    Cory = correlationdirprinc(R, 2)[1]
    # Tracé des flèches et de leur label
    for i in range(np.shape(R)[1]):
        ax4.arrow(0, 0, Corx[i], Cory[i], width=0.02,
                  length_includes_head=True, color='royalblue')
        ax4.annotate(text=labelscolonne[i], xy=(
            Corx[i], Cory[i]),  fontsize=12)

    plt.show()
    return


def ACP3D(R, labelsligne, labelscolonne):
    """
    Parameters
    ----------
    R : TYPE
        DESCRIPTION.
    labelsligne : TYPE
        DESCRIPTION.
    labelscolonne : TYPE
        DESCRIPTION.
    Returns
    -------
    Renvoie deux figures :
            Fig 1 – Valeurs des variances et pourcentages explicatifs de celles-ci pour les composantes
            Fig 2 – Représentation des données dans l’espace (Y1, Y2, Y3) et le cercle de corrélation dans ce même espace (3D)
    """

    # _______________Figure 3 Valeurs des variances et pourcentage explicatif de celles-ci pour les composantes principales_______________
    # Mise en page de la fenetre matplotlib
    # séparation  de la fenetre en deux pour afficher deux graphiques
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle('Figure 1 – Valeurs des variances et pourcentages explicatifs de celles-ci pour les composantes principales',
                 fontsize=12)  # Titre général de la fenêtre
    # Tracé de la Variance des composantes principales

    X = centre_red(R)  # La matrice  des données est centrée et réduite
    # Décomposition SVD de la matrice X cnetrée réduite
    u, s, vt = np.linalg.svd(X)
    v = vt.T  # Transposiiton du vecteur vt provenant de la décompositon SVD. v Vecteur

    # On récupère les dimensions de X la matrice centrée réduite sur laquelle on travaille (Même dimension que R)
    m, k = np.shape(X)
    # On crée une matrice Yk de 2 colonnes. Elle nous servira à afficher les variances des composantes
    Yk = np.zeros(shape=(k, 2), dtype=object)
    for i in range(k):  # On remplit cette matrice Yk
        vk = v[:, i].reshape(len(v), 1)
        # La première colonne contient le label Yk
        Yk[i, 0] = '$Y_{}$'.format(i+1)
        # La seconde colonne contient la valeur de la variance correspondante
        Yk[i, 1] = Variance(X@vk)
    Yksum = 0
    for i in range(len(Yk)):
        Yksum += Yk[i, 1]
    ind = Yk[:, 0]  # Liste contenant les labels de la colonne 0 de la matrice Yk créée précédement. Elle servira pour les labels des différentes variances
    # Création d'une nouvelle liste qui ne contient que les valeurs des différentes variances pour tracer le graphique en barres
    Ykvalues = list([])
    for i in range(k):  # On récupère les valeurs de variance contenue dans Yk et on les reconvertit du format object vers float
        Ykvali = float(Yk[i, 1])
        Ykvalues.append(Ykvali)
    labelsYkvalues = sorted(Ykvalues, reverse=True)

    # Création du graphique en barres
    ax1.bar(ind, Ykvalues, width=0.8, color='royalblue',)
    ax1.set_xlabel('Composantes', fontsize=9)
    ax1.set_ylabel('$Var(Y_{k})$', fontsize=9)
    ax1.set_title('Variances des composantes principales')

    # On place les valeurs de chacune des barres au dessus de celles-ci
    for index, value in enumerate(labelsYkvalues):
        ax1.text(index-0.2, value+0.04, str(round(value, 6)))

    # Création du graphique représentant le pourcentage de l’explication de la variance de chaque k−composantes principales

    # Calcul du pourcentage
    Ykpercent = []
    for i in range(len(Ykvalues)):
        Ykpercent.append((Ykvalues[i]*100)/Yksum)

    ax2.pie(Ykpercent, labels=ind, autopct=lambda Ykpercent: str(round(
        Ykpercent, 1)) + '%', shadow=True)  # "autopct" permet d'ajouter sur chacune des parts
    # la proportion en % qu'elle représente
    ax2.set_title('Participation à la variance totale')

    # _______________Figure 4 Représentation des données dans l’espace (Y1, Y2, Y3) et le cercle de corrélation dans ce même espace
    # Mise en page de la fenetre matplotlib
    fig2 = plt.figure()
    plt.suptitle('Figure 2 – Représentation des données dans l’espace (Y1, Y2, Y3) et le cercle de corrélations dans ce même espace', fontsize=12)

    # Création du graphe en 3D qui représente la matrice de sortie de approx(R,3)
    ax3 = fig2.add_subplot(1, 2, 1, projection='3d')
    ax3.set_title('Analyses en composantes principales pour k=3')
    ax3.set_xlabel('$Y_{1}$')
    ax3.set_ylabel('$Y_{2}$')
    ax3.set_zlabel('$Y_{3}$')

    # Utilisation de la méthode découverte en Ma 313 pour colorier chacun des points du nuage selon la valeur de leur labelsligne
    # Création d'un dictionnaire contenant chacun des labels ligne et assignant une couleur aléatoire à ces labels.
    Dico = {}
    n = len(set(labelsligne))
    lbl = labelsligne
    for i in labelsligne:
        if i not in Dico:
            color = "%06x" % random.randint(0, 0xFFFFFF)
            Dico[i] = color

    # Tracé des points et de leur label
    App = (approx(R, 3))
    m, n = np.shape(R)
    for i in range(m):
        ax3.scatter(App[i, 0], App[i, 1], App[i, 2], color='#' + Dico[lbl[i]])
    for i, txt in enumerate(lbl):
        ax3.text(App[i, 0], App[i, 1], App[i, 2],  txt,
                 size=8, color='#' + Dico[lbl[i]])

    # Création du graphe de la matrice de sortie de la fonction correlationdirprinc(R,3)
    ax4 = fig2.add_subplot(1, 2, 2, projection='3d')
    ax4.set_title('Cercles de corrélation et ses projections')
    ax4.set_xlabel('$Y_{1}$')
    ax4.set_ylabel('$Y_{2}$')
    ax4.set_zlabel('$Y_{3}$')
    xyzlim = 1.3
    ax4.set_xlim(-xyzlim, xyzlim)
    ax4.set_ylim(-xyzlim, xyzlim)
    ax4.set_zlim(-xyzlim, xyzlim)

    # Sphère de correlation
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax4.plot_surface(x, y, z, color='royalblue', alpha=0.3)

    # Projection des cercles de correlation
    p1 = Circle((0, 0), 1, fill=False)
    ax4.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=-xyzlim, zdir="x")
    p2 = Circle((0, 0), 1, fill=False)
    ax4.add_patch(p2)
    art3d.pathpatch_2d_to_3d(p2, z=xyzlim, zdir="y")
    p3 = Circle((0, 0), 1, fill=False)
    ax4.add_patch(p3)
    art3d.pathpatch_2d_to_3d(p3, z=-xyzlim, zdir="z")

    # Utilisation de la fonction correlationdirprinc pour récupérer les coordonnées de chacune des flèches à tracer dans le cercle de correlation
    Corx = correlationdirprinc(R, 3)[0]
    Cory = correlationdirprinc(R, 3)[1]
    Corz = correlationdirprinc(R, 3)[2]
    # Tracé des flèches et leurs labels dans la sphère de correlation
    for i in range(np.shape(R)[1]):
        ax4.quiver(0, 0, 0, Corx[i], Cory[i], Corz[i], color='royalblue')
    for i, txt in enumerate(labelscolonne):
        ax4.text(Corx[i], Cory[i], Corz[i], txt)

    # Tracé des projections flèches et leurs labels sur les cercles de projection de la sphère de correlation
    # Flèches
    for i in range(np.shape(R)[1]):
        ax4.quiver(0, 0, -xyzlim, Corx[i], Cory[i], 0, color='red')
    for i in range(np.shape(R)[1]):
        ax4.quiver(-xyzlim, 0, 0, 0, Cory[i], Corz[i], color='red')
    for i in range(np.shape(R)[1]):
        ax4.quiver(0, xyzlim, 0, Corx[i], 0, Corz[i], color='red')
    # Labels
    for i, txt in enumerate(labelscolonne):
        ax4.text(Corx[i], Cory[i], -xyzlim, txt)
    for i, txt in enumerate(labelscolonne):
        ax4.text(Corx[i], xyzlim, Corz[i], txt)
    for i, txt in enumerate(labelscolonne):
        ax4.text(-xyzlim, Cory[i], Corz[i], txt)

    plt.show()
    return


def ACP(R, labelsligne, labelscolonne, k=0, epsilon=10**-1):
    """
    Parameters
    ----------
    R : TYPE
        DESCRIPTION.
    labelsligne : TYPE
        DESCRIPTION.
    labelscolonne : TYPE
        DESCRIPTION.
    k : TYPE, optional
        DESCRIPTION. The default is 0.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 10**-1.
    Returns
    -------
    None.
    """
    "faire la relation de Kaiser avec k et espilon"

    # _______________Figure 1 Valeurs des variances et pourcentages explicatifs de celles-ci pour les composantes principales_______________
    # Mise en page de la fenetre matplotlib
    # séparation  de la fentre en deux pour afficher deux graphiques
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle('Figure 1 – Valeurs des variances et pourcentage explicatif de celles-ci pour les composantes principales',
                 fontsize=12)  # Titre général de la fenêtre
    # Tracé de la Variance des composantes principales

    X = centre_red(R)  # La matrice  des données est centrée et réduite
    # Décomposition SVD de la matrice X centrée réduite
    u, s, vt = np.linalg.svd(X)
    v = vt.T  # Transposiiton du vecteur vt provenant de la décompositon SVD. v Vecteur

    # On récupère les dimensions de X la matrice centrée réduite sur laquelle on travaille (Même dimension que R)
    m, n = np.shape(X)
    # On crée une matrice Yk de 2 colonnes. Elle nous servira à afficher les variances des composantes
    Yk = np.zeros(shape=(n, 2), dtype=object)
    for i in range(n):  # On remplit cette matrice Yk
        vk = v[:, i].reshape(len(v), 1)
        # La première colonne contient le label Yk
        Yk[i, 0] = '$Y_{}$'.format(i+1)
        # La seconde colonne contient la valeur de la variance correspondante
        Yk[i, 1] = Variance(X@vk)
    Yksum = 0
    for i in range(len(Yk)):
        Yksum += Yk[i, 1]
    ind = Yk[:, 0]  # Liste contenant les labels de la colonne 0 de la matrice Yk créée précédemment. Elle servira pour les labels des différentes variances
    # Création d'une nouvelle liste qui ne contient que les valeurs des différentes variances pour tracer le graphique en barres
    Ykvalues = list([])
    for i in range(n):  # On récupère les valeurs de variance contenue dans Yk et on les reconvertit du format object vers float
        Ykvali = float(Yk[i, 1])
        Ykvalues.append(Ykvali)
    labelsYkvalues = sorted(Ykvalues, reverse=True)

    # Création du graphique en barres
    ax1.bar(ind, Ykvalues, width=0.8, color='royalblue',)
    ax1.set_xlabel('Composantes', fontsize=9)
    ax1.set_ylabel('$Var(Y_{k})$', fontsize=9)
    ax1.set_title('Variances des composantes principales')

    # On place les valeurs de chacune des barres au dessus de celles-ci
    for index, value in enumerate(labelsYkvalues):
        ax1.text(index-0.2, value+0.04, str(round(value, 6)))

    # Création du graphique représentant le pourcentage de l’explication de la variance de chaque k−composantes principales

    # Calcul du pourcentage
    Ykpercent = []
    for i in range(len(Ykvalues)):
        Ykpercent.append((Ykvalues[i] * 100) / Yksum)

    ax2.pie(Ykpercent, labels=ind, autopct=lambda Ykpercent: str(round(
        Ykpercent, 1)) + '%', shadow=True)  # "autopct" permet d'ajouter sur chacune des parts
    # la proportion en % qu'elle représente
    ax2.set_title('Participation à la variance totale')

    # _______________Représentation de la matrice de corrélation entre les Yi et les Xj sur Iris (k=2 d’après la règle de Kaiser avec Epsilon = 10^−1)_____
    fig2, ax = plt.subplots()
    plt.suptitle('Figure 2 – Représentation de la matrice de corrélation entre les Yi et les Xj (k=2 d’après la règle de Kaiser avec Epsilon = 10\u207B\u00b9)', fontsize=12)
    plt.imshow(correlationdirprinc(R, k))
    plt.subplots_adjust(bottom=0, right=0.8, top=1)
    bar = plt.axes([0.85, 0.1, 0.025, 0.7])
    txt = ['A', 'B']
    ax.set_ylabel(ylabel=txt, loc='top')
    ax.set_xlabel('XLabel', loc='left')
    plt.colorbar(cax=bar)

    return()

# %%


def etape1(A, k):  # On crée uj0 qui contiendra k vecteur random de A
    m, n = np.shape(A)
    nb_random = np.random.randint(m, size=(1, k))[0]
    uj0 = []
    for i in range(k):
        uj0.insert(i, A[nb_random[i], :])  # Ajout de la ligen random dans u0j

    return(np.asarray(uj0))  # OK


def etape2(A, u0j, k, j):  # On calcul maintenant nos partitions avec la condition imposé dans le sujet
    m = np.shape(A)[0]
    Sj1 = []
    p = []
    for i in range(k):
        if i != j:
            p.append(i)

    for i in range(m):

        norme1 = np.linalg.norm(A[i, :]-u0j[j], 2)
        if norme1 <= np.linalg.norm(A[i, :]-u0j[p[0]], 2):
            if norme1 <= np.linalg.norm(A[i, :]-u0j[p[1]], 2):
                Sj1.insert(i, A[i, :])

    return(np.asarray(Sj1))  # OK
# OK


def etape3(Sj, k, g):
    u1 = []
    sum = np.zeros((np.shape(Sj[0])))
    for j in range(k):
        for ai in Sj:
            sum = sum+ai  # somme des vects de Sj
        # moyenne des vecteurs de Sj inséré dasn u1
        u1.insert(j, sum/np.shape(Sj)[0])
    print("Matrice u1{} :\n".format(g), np.asarray(u1))
    return(np.asarray(u1))  # OK


def etape45(A, k, epsilon):  # étape 4 et 5 qui utilise les fonctions précedentes
    f = 1
    uj0 = etape1(A, k)
    ujk = []
    Sj1 = etape2(A, uj0, k, f-1)
    uj1 = etape3(Sj1, k, 0)
    u = [uj0, uj1]
    S = [Sj1]
    # itération de l'étape 2 et 3 f fois et renvoie S
    while np.linalg.norm(u[f-1]-u[f]) > epsilon and f < k:

        Sjk = etape2(A, uj0, k, f)
        ujk = etape3(Sjk, k, f)
        f = f+1
        u.insert(f, ujk)
        S.insert(f, Sjk)

        u.insert(f, ujk)
    return(S)


def Kmoy(A, k=0, epsilon=10**-1):
    """
    Parameters
    ----------
    A : Matrice d'entrée
    k : int >= 2. The default is 0
    epsilon : float. The default is 10**-1.
    Returns
    -------
    S
    """
    S = etape45(A, k, epsilon)

    return(S)


k = 3
A = approx(Kmoy(E, k=3, epsilon=10**-1)[0], 2)
A1 = approx(Kmoy(E, k=3, epsilon=10**-1)[1], 2)
A2 = approx(Kmoy(E, k=3, epsilon=10**-1)[2], 2)
for i in range(k):
    plt.scatter(A1[:, 0], A1[:, 1])
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(A2[:, 0], A2[:, 1])
plt.show()
# %%
