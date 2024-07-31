import numpy as np
from math import sqrt
from functools import partial

from jax import jit
import jax.numpy as jnp

def WW_transform(L,M,rng_func,H2,H4, ezsol = True):
    Gr = [1,-1]
    groups = [[1,],Gr]

    @jit
    def jnp_trans(bf,H2,H4):
        temp2 = jnp.einsum("xi,ij,yj->xy",bf,H2,bf)
        temp4 = jnp.einsum("xi,yj,ijkl,zk,ql->xyzq",bf,bf,H4,bf,bf)
        return temp2,temp4

    def generate_window_functions(N,M):
        L = N*M
        nlist = np.arange(-round(L/2)+1,round(L/2))
        f = lambda x: np.where(nlist == x)[0][0]
        N4 = round(N/4)
        N2 = round(N/2)
        G = np.zeros(L)

        G[f(0)] = sqrt(2*M/L)
        N4t = N4
        if (N/4).is_integer():
            G[f(N4)] = sqrt(M/L)
            N4t -= 1

        # Populate momentum space
        if N4 >= 2:
            G[f(1):f(N4)] = (rng_func(size = N4-1))*sqrt(2*M/L)

        # Orthogonality condition
        G[f(N4)+1:f(N2)] = list(map(lambda gt:sqrt(2*M/L - pow(abs(gt),2)),G[f(1):f(N4t)+1]))[::-1]

        # Apply G(p) = G(-p)
        G[f(-N2)+1:f(0)] = G[f(1):f(N2)][::-1]

        # FT
        g = np.zeros(L,dtype=complex)
        for j in range(L):
            g[j] = np.sum(list(map(lambda i: np.exp(complex(0,j*2*np.pi*i/L))*G[f(i)], nlist)))/sqrt(L)

        return g

    def Hk(k):
        return groups[int(abs(k) in [0,M])]

    # exponential phase factor
    def EPF(m,k,alpha):
        if (m+k)%2 == 0:
            return 1.
        return complex(0,-alpha)

    N = round(L/M)
    g = generate_window_functions(N,M)
    sg1 = np.zeros((N,M+1,L),dtype =complex)
    sg2 = np.zeros((N,M+1,L),dtype =complex)
    for m in range(N):
        for k in range(M+1):
            f_sg1 = lambda j: np.exp(complex(0, np.pi*k*j/M))*g[(j-m*M)%L]
            f_sg2 = lambda j: np.exp(complex(0, -np.pi*k*j/M))*g[(j-m*M)%L]

            sg1[m,k] = list(map(f_sg1,range(L)))
            sg2[m,k] = list(map(f_sg2,range(L)))

            # PROBLEM A CHEAP SOLUTION #
            if ezsol and m%2 == 1 and (m+k)%2 == 0:
                sg1[m,k] = -np.conjugate(sg1[m,k])
                sg2[m,k] = -np.conjugate(sg2[m,k])

    bf = np.zeros((L,L),dtype = complex)
    for m in range(N):
        mkrange = range(M+1)
        krange  = range(M+1)
        if m%2 == 1:
            mkrange = range(M+1,2*M)
            krange  = range(1,M)
        for mk,k in zip(mkrange,krange):
            for j in range(L):
                # can make lambda function here; no gain in performance though
                for a,sg in zip(Gr,(sg1,sg2)):
                    bf[mk+(m//2)*(2*M),j] += EPF(m,k,a)*sg[m,k,j]

                bf[mk+(m//2)*(2*M),j] /= sqrt(len(Gr)*len(Hk(k)))

    bf = bf.real

    temp2, temp4 = jnp_trans(jnp.array(bf),jnp.array(H2),jnp.array(H4))
    temp2 = np.array(temp2)
    temp4 = np.array(temp4)
    temp2 = np.where(abs(temp2)>1.e-15,temp2,0.)
    temp4 = np.where(abs(temp4)>1.e-15,temp4,0.)

    return temp2,temp4

def diagonal_transform(H2,H4):
    H2_t, P = np.linalg.eigh(H2)
    PT = P.transpose()
    H4_t = np.einsum("ijkl,mi,nj,kp,lr->mnpr",H4,P,P,PT,PT)
    return np.diag(H2_t),H4_t

def no_transform(H2,H4):
    return H2,H4

def pdf_gauss_cheat(M,size = 1):
    x = np.linspace(1,M*sqrt(np.log(2)),size+1)[:-1]
    p_gauss = lambda x: np.exp(-x**2/(2*M**2))
    return np.array(list(map(p_gauss,x)))

class basis:
    def __init__(self,option,**kwargs):
        self.kwargs = kwargs
        self.basis_names = {
            0: 'None',
            1: 'Diagonal',
            2: 'Wilson-Wannier'

        }
        self.option = option
        try:
            L = int(kwargs["L"])  # Chain length
            self.L = L
        except KeyError:
            raise KeyError("basis (init-WW): Wilson-Wannier transformation requires keys 'L'")
        if option == 0:
            self.transform_func = no_transform
        elif option == 1:
            self.transform_func = diagonal_transform
        elif option == 2:
            try:
                M = int(kwargs["M"]) # Real space shift length
            except KeyError:
                raise KeyError("basis (init-WW): Wilson-Wannier transformation requires keys 'M'")
            except:
                raise KeyError("basis (init-WW): Unknown error occured.")
            if M%2 != 0:
                raise ValueError("basis (init-WW): Key 'M' must be an even integer.")
            if int(L/M) % 2 != 0:
                raise ValueError("basis (init-WW): L/M must be an even integer.")
            pdf_gauss = partial(pdf_gauss_cheat,M)
            self.M = M
            self.transform_func = partial(WW_transform,L,M,pdf_gauss)
        else:
            raise ValueError("basis (init): Incorrect option.")

    def transform(self,H2,H4):
        return self.transform_func(H2,H4)
    
    def __mul__(self,other:tuple):
        if not all(isinstance(tensor,np.ndarray) for tensor in other):
            raise Exception("Incorrect type")
        
        for idx,tensor in enumerate(other):
            if tensor.shape != tuple([self.L for _ in range((idx+1)*2)]):
                raise Exception("Incorrect shapes")
        return self.transform_func(*other)
    
    def __str__(self):
        extra = ''
        if self.option == 2:
            extra += f"_M{self.M}"
        return f"{self.basis_names[self.option]}{extra}"
