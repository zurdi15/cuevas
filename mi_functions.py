from sympy import *
import numpy as np

#ALGORITMO PARA LOS POLINOMIOS DE CHEBYSCHEV
def poli_cheb (N):
    x=symbols('x')
    T0=1
    T1=x
    Cheb=[]
    Cheb.append(T0)
    Cheb.append(T1)

    for i in range (N-1):
        Tf=simplify(2*x*T1-T0)
        Cheb.append(Tf)
        T0=T1
        T1=Tf

    return Cheb

#ALGORITMO POLINOMIOS INTERPOLADORES DE LAGRANGE
def Lagrange_Cardinales(N,Absc):
    x=symbols('x')
    C=[]
    L=1
    
    for i in range(N+1):
        for j in range (N+1):
            if(j!=i):
             L=L*(x-Absc[j])/(Absc[i]-Absc[j])
            else:
                continue

        C.append(simplify(L)) 
        L=1  
    
    return C

#Algoritmo para calcular errores:
def Error_Tot (fun_cal,fun_teo,a,b):
    x=symbols('x')
    U=abs(fun_cal-fun_teo)
    ET=integrate(U,(x,a,b))
    return ET

def Abscisas_Gauss_Chebyshev (Ord):
        Abs_GC=[]
        for i in range(Ord+1):
            Abs_GC.append(float(-np.cos((2*(i)+1)*np.pi/(2*(Ord+1)))))
        
        return Abs_GC

def Abscisas_Gauss_Lobatto (Ord):
    x_GL=[]
    
    for i in range (Ord+1):
        x_GL.append(float(-np.cos((i)*np.pi/Ord)))
    

    return x_GL

def Derivadas (Cheb,ord_deriv):
    x=symbols('x')
    D=[]
    for i in range (len(Cheb)):    
        D.append(simplify(diff(Cheb[i],x,ord_deriv)))

    return D   

def Matriz_Colocacion_Ort(Pol,N,Absc,R):
    x=symbols('x')
    #Creamos un nuevo vector con los polinomios pero ahora en forma numérica
    Nche=[]
    for i in range (N+1):
        Nche.append(lambdify(x,Pol[i]))
    #Metemos en la matriz M los valores de los polinomios en las condiciones de contorno:
    M=[]


    for i in range (N+1):
        fila=[]
        for j in range (N+1):
            Aux=lambdify(x,R[j],'numpy')
            fila.append(float(Aux(Absc[i])))
        M.append(fila)
    
    return M


def Lagrange_Ordenadas (Absc,fun):
    x=symbols('x')
    fun_num=lambdify(x,fun,'numpy')
    Ordenadas=[]
    for i in range (len(Absc)):
        Ordenadas.append(fun_num(Absc[i]))

    #Polinomio final:
    return Ordenadas

def Matriz_Colocacion_Ort_Lagrange(Pol,N,Absc,R):
    x=symbols('x')
    #Creamos un nuevo vector con los polinomios pero ahora en forma numérica
    Nche=[]
    for i in range (N+1):
        Nche.append(lambdify(x,Pol[i]))
    #Metemos en la matriz M los valores de los polinomios en las condiciones de contorno:
    M=[]

    for i in range (N+1):
        fila=[]
        for j in range (N+1):
            Aux=lambdify(x,R[j],'numpy')
            fila.append(float(Aux(Absc[i])))
        M.append(fila)
    
    return M

def Pesos(w,C):
    x=symbols('x')
    W=[]
    for i in range (len(C)):
        W.append(float(integrate(C[i]*w,(x,-1,1))))

    return W

def Matrix_Transformation(W,T,Abs):
    x=symbols('x')
    M=[]
    for i in range(len(W)):
        fila=[]
        for j in range (len(Abs)):
            Aux=lambdify(x,T[i],'numpy')
            fila.append(W[j]*Aux(Abs[j])/float(integrate(T[i]*T[i]*1/sqrt(1-x**2),(x,-1,1))))
        M.append(fila)
    
    return M

def Fourier_trigonometrico(Orden):
    th=symbols('th')
    F=[]
    for i in range (Orden):
        if i<floor(Orden/2)+1:
            F.append(cos(i*th))
        else:
            F.append(sin((i-floor(Orden/2))*th))
    
    return F