import matplotlib.pyplot as plt

import sympy as sp
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
import mi_functions as mf
import pandas as pd

N=3#Orden polinomial de cada subdominio
k=1.5
[x,r,s,t,y]=sp.symbols('x r s t y')
T=mf.poli_cheb(N)

T_Abs=T[N-1]

print(T_Abs)
print('\n')
#Generamos abscisas según cuadratura de Lobatto (r=2*y+1 y(-1,0),s=2*y-1 y(0,1)):
coeffs = sp.Poly(T_Abs, x).all_coeffs()
Absc1=[-1]
pc=np.roots(coeffs)

for i in range (N-1):
    Absc1.append(float(pc[i]))
Absc1.append(1)
Absc=np.sort(Absc1)

#Matriz A multiplica vector derivadas
A=[]
for i in range (2*N+2):
    fila=[]
    for j in range (2*N+2):
        if  i<N and j<N+1:
            Aux=sp.lambdify(x,T[j])
            fila.append(float(Aux(Absc[i])))
        if i<N and j>=N+1:
            fila.append(0)
        if i==N and j<N+1:
            Aux=sp.diff(T[j],x,1)
            Aux2=sp.lambdify(x,Aux)
            fila.append(float(Aux2(Absc[N])))
        if i==N and j>=N+1:
            Aux=sp.diff(T[j-N-1],x,1)
            Aux2=sp.lambdify(x,Aux)
            fila.append(float(-k*Aux2(Absc[0])))
        if i==N+1 and j<N+1:
            Aux2=sp.lambdify(x,T[j])
            fila.append(float(Aux2(Absc[N])))
        if i==N+1 and j>=N+1:
            Aux2=sp.lambdify(x,T[j-N-1])
            fila.append(float(Aux2(Absc[0])))
        if i>N+1 and j<N+1:
            fila.append(0)
        if i>N+1 and j>=N+1:
            Aux2=sp.lambdify(x,T[j-N-1])
            fila.append(float(Aux2(Absc[i-N-1])))
    A.append(fila)


print('\n')
#MATRIZ B Multiplica vector sin derivar

B=[]
for i in range (2*N+2):
    fila=[]
    for j in range (2*N+2):
        if i==0 or i==2*N+1:
            fila.append(0)
        if i>0 and i<N and j<N+1:
            Aux=sp.diff(T[j],x,2)
            Aux2=sp.lambdify(x,Aux)
            fila.append(float(Aux2(Absc[i])))
        if i>0 and i<N and j>=N+1:   
            fila.append(0)
        if i==N or i==N+1:
            fila.append(0)
        if i>N+1 and i<2*N+1:
            if j<N+1:
                fila.append(0)
            else:
                Aux=sp.diff(T[j-N-1],x,2)
                Aux2=sp.lambdify(x,Aux)
                fila.append(float(Aux2(Absc[i-N-1])))
    B.append(fila)


#Matriz C:
C=np.dot(np.linalg.inv(A),B)


coef_exp,V=np.linalg.eig(C)

D=[]
for i in range (2*N+2):
    fila=[]
    for j in range (2*N+2):
        if i==j:
            fila.append(sp.exp(coef_exp[i]*t))
        else:
            fila.append(0)
    D.append(fila)



#Construimos la matriz de condiciones iniciales:
A_ini=[]
for i in range (2*N+2):
    fila=[]
    for j in range (2*N+2):
        if i<N and j<N+1:
            Aux=sp.lambdify(x,T[j])
            fila.append(float(Aux(Absc[i])))
        if i<N and j>=N+1:
            fila.append(0)
        if i==N and j<N+1:
            Aux=sp.lambdify(x,T[j])
            fila.append(float(Aux(1)))
        if i==N and j>=N+1:
            Aux=sp.lambdify(x,T[j-N-1])
            fila.append(float(-Aux(-1)))
        if i==N+1 and j<N+1:
            Aux=sp.diff(T[j],x,1)
            Aux2=sp.lambdify(x,Aux)
            fila.append(float(Aux2(1)))
        if i==N+1 and j>=N+1:
            Aux=sp.diff(T[j-N-1],x,1)
            Aux2=sp.lambdify(x,Aux)
            fila.append(float(-k*Aux2(-1)))
        if i>N+1 and j<N+1:
            fila.append(0)
        if i>N+1 and j>=N+1:
            Aux=sp.lambdify(x,T[j-N-1])
            fila.append(float(Aux(Absc[i-N-1])))
    A_ini.append(fila)


#Creamos vector residuo condiciones iniciales
b=[]
for i in range(2*N+2):
    if i<=N+1:
        b.append(0)
    else:
        b.append(1)
V_inv=np.linalg.inv(V)


Sol_coef_ini=np.dot(np.linalg.inv(A_ini),b)

Mat_aux=np.dot(V,D)


Mat_aux_2=np.dot(Mat_aux,V_inv)

print('\n')

Sol_Final=np.dot(Mat_aux_2,Sol_coef_ini)

P_1=np.dot(T,Sol_Final[:N+1])
P_2=np.dot(T,Sol_Final[N+1:])
P_1_FINAL=sp.simplify(P_1.subs(x,2*y+1))
P_2_FINAL=sp.simplify(P_2.subs(x,2*y-1))
#Integramos los polinomios de Chebyshev en el intervalo y(-1,0) y(0,1)
print(P_1_FINAL)
print('\n')
print(P_2_FINAL)

SOL_EXAC_EST_1=k/(1+k)*(y+1)
SOL_EXAC_EST_2=1/(1+k)*(y+k)



#Cheb_int=[]
#for i in range (N+1):
    #aux=sp.integrate(T[i],(x,-1,1))
    #if aux<0:
   #     Cheb_int.append(-aux)
    #else:
     #   Cheb_int.append(aux)


#P_1_INT=np.dot(Cheb_int,Sol_Final[:N+1])
#P_2_INT=np.dot(Cheb_int,Sol_Final[N+1:])

#Sol_exac_int_1=sp.integrate(SOL_EXAC_EST_1,(y,-1,0))
#Sol_exac_int_2=sp.integrate(SOL_EXAC_EST_2,(y,0,1))

# print("\n\n\n-----TIPOS DE DATOS-----")
# print(P1F)
# print(type(P1F))
# print(SOL_EXAC_EST_1)
# print(type(SOL_EXAC_EST_1))
# print("-----TIPOS DE DATOS-----\n\n\n")

# Compute errors as SymPy expressions first
ERR1 = P_1_FINAL - SOL_EXAC_EST_1
ERR2 = P_2_FINAL - SOL_EXAC_EST_2

# Integrate the absolute error over the domains
ER_1 = sp.integrate(sp.Abs(ERR1), (y, -1, 0))
ER_2 = sp.integrate(sp.Abs(ERR2), (y, 0, 1))

# Lambdify the integrated error for plotting
ER_1_func = sp.lambdify(t, ER_1, modules=['numpy', 'scipy'])
ER_2_func = sp.lambdify(t, ER_2, modules=['numpy', 'scipy'])

# Vectorize to handle arrays (needed because of numerical integration in lambdified functions)
ER_1_vec = np.vectorize(ER_1_func)
ER_2_vec = np.vectorize(ER_2_func)

print("ER_1:", ER_1)
print("ER_2:", ER_2)

# Use fewer time points for faster computation
time = np.linspace(0, 5, 20)

print(f"\nComputing errors for {len(time)} time points...")
print("This may take a moment due to numerical integration...")

# Compute errors
error1_values = ER_1_vec(time)
print("Domain 1 errors computed")

error2_values = ER_2_vec(time)
print("Domain 2 errors computed")

fig, ax = plt.subplots()

ax.set_title('Error  $E(N=2,k=1.5)$')
ax.set_xlabel('t')
ax.plot(time, error1_values, 'o-', label="Error Domain 1")
ax.plot(time, error2_values, 's-', label="Error Domain 2")
ax.plot(time, error1_values + error2_values, '^-', label="Error Total")


plt.legend(loc='best')
plt.grid(True)
plt.show()