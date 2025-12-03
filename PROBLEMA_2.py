import numpy as np
from sympy import *
import mi_functions as mf
x,th=symbols ('x th')
N=3
M=3
#generar poli chebyshev,
#generar funciones Fourier, 
#serán dos listas de tamaño N+1 y M, la sumamos individualmente y el resultado se multiplica, dándo lugar a una única función U.
Cheb=mf.poli_cheb(N)
Four=mf.Fourier_trigonometrico(M)

def f(ang):
    if ang<=np.pi:
        return ang
    else:
        return(2*np.pi-ang)
G=[] #matriz generatriz
#Aplicamos la EDO sobre la función U 
 #Función Resiudo
for i in range(N*M):
    fila=[]
    for j in range (N+1):
        for k in range (M):
            R=simplify(diff(Cheb[j],x,2)*Four[k]+1/(x+1)*diff(Cheb[j],x,1)*Four[k]+1/((x+1)**2)*diff(Four[k],th,2)*Cheb[j])
            fila.append(R)
    G.append(fila)
#Añadimos Condiciones  de Contorno
for i in range(M):
    fila=[]
    for j in range (N+1):
        for k in range (M):
            BC=simplify(Cheb[j]*Four[k])
            fila.append(BC)
    G.append(fila)
#Generamos las abscisas:
Rc=[]
for i in range (N+1):
    aux=-cos((i+1)*np.pi/(N+1))
    Rc.append(aux)
Thc=[]
for i in range (M):
    aux=2*np.pi*i/M
    Thc.append(aux)

#Ahora pasamos a numérico todas las componentes de la matriz, y vamos sustituyendo los valores de las abscisas
G_num=[]

for i in range(N):
    for w in range(M):
        fila=[]
        for j in range (N+1):
            for k in range (M):
                AUX1=lambdify((x,th),G[i*M+w][j*M+k],'numpy')
                a1=float(Rc[i])
                b1=float(Thc[w])
                # result1=AUX1(a1,b1)
                result1=float(AUX1(a1,b1))
                fila.append(result1)
        G_num.append(fila)
#Pasamos a numérico las condiciones de contorno:

    for w in range(M):
        fila=[]
        for j in range (N+1):
            for k in range (M):
                AUX2=lambdify((x,th),G[N*M+w][j*M+k],'numpy')     
                result2=float(AUX2(float(Rc[N]),float(Thc[w])))
                fila.append(result2)
        G_num.append(fila)

for fila in G_num:
    print(fila)

#Vector b:
b=[]
for i in range ((N+1)*M):
    if i>N*M-1:
        for j in range (M):
            b.append(f(Thc[j]))
        break
    else:
        b.append(0)

# a=G_num[0][1]*G_num[0][4]
# print(a)