# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget

# %%


def f(x):
    return (np.pi/2)*np.sin(np.pi*x/2)


# %%
def Newton_Cotes4(f, a, b, n):
    S = 0
    W = [7/90, 16/45, 2/15, 16/45, 7/90]
    x = np.linspace(a, b, n)
    for j in range(1, n):
        for i in range(len(W)):
            S += W[i]*f(x[j])*(x[j]-x[j-1])
    return S


# %%
n = np.linspace(50, 1000, 20)
a = 0
b = 1
Sol_exact = 1
Ef = []
for i in range(len(n)):
    Sol_appro = Newton_Cotes4(f, 0, 1, int(n[i]))
    Ef.append(Sol_appro-Sol_exact)
print(Ef)
plt.figure()
plt.title("Erreur relative tracer log.")
plt.loglog(n, Ef, 'r')
plt.show()

# %%
%matplotlib widget


def Newton_Cotes4(f, a, b, n, k):
    S = 0
    W = [7/90, 16/45, 2/15, 16/45, 7/90]
    x = np.linspace(a, b, n)
    for j in range(1, n):
        for i in range(k+1):
            S += W[i]*f(x[j])*(x[j]-x[j-1])
    return S


def A(r):
    return (c**2)*(1+K/r)


def B(r):
    return (1+K/r)**(-1)


def f(x):
    r0 = 0.0046547454
    if x == r0:
        return 0
    else:
        return(1/x)*(B(x)/(((A(r0)*x**2)/(A(x)*r0**2))-1))**0.5


def Int_phi(r, r0, N, k):
    return Newton_Cotes4(f, r0, r, N, k)


max = 50

G = 4*np.pi**2
c = 63239.7263
M = 1
K = -2*G*M/c**2

plt.figure()
plt.title("trajectoire d'un rayon de lumière \n passant à proximité du Soleil")
r0 = 0.0046547454
n = r0
r = r0+n
while (max+1)*r0 > r:
    Sol_appro = Int_phi(r, r0, 1000, 4)
    Photon = plt.scatter(r*np.cos(Sol_appro), r *
                         np.sin(Sol_appro), color="red", marker=".")
    n += r0
    r = r0+n
Soleil = plt.scatter(r0*np.cos(0), r0*np.sin(0),
                     color="b", marker="o", label="Soleil")
plt.xlabel('x')
plt.ylabel('y')
plt.legend((Soleil, Photon), ("Soleil", "Photon"),
           scatterpoints=1, loc='upper left', fontsize=10)
plt.show()
