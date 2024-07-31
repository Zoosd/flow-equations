import numpy as np

DTYPE = float

# c^\dag_i c^\dag_j c_k c_l
def nearest_neighbour_interactions(L:int,Delta:float) -> np.ndarray:
    """ The quartic part of a Hamiltonian containing nearest neighbour interactions.

    Args:
        L (int): Number of sites in the chain.
        Delta (float): Interaction strength.

    Returns:
        H4: The quartic part of a Hamiltonian containing nearest neighbour interactions as a rank-4 tensor.
    """
    H4 = np.zeros((L,L,L,L),dtype=DTYPE)
    for i in range(L):  
        j = (i+1)%L
        H4[i, j, i, j] = -Delta/4.0
        H4[i, j, j, i] = Delta/4.0
        H4[j, i, i, j] = Delta/4.0
        H4[j, i, j, i] = -Delta/4.0

    return H4

def disordered_potential(L:int,W:float,R:int):
    h = np.array([np.diag(np.random.uniform(low=-W, high=W, size=L).astype(DTYPE)) for _ in range(R)])
    return h.reshape((R,L,L))


def linear_potential(L:int,F:float):
    return np.diag((F*np.arange(L)).astype(DTYPE))

def left_right_hopping(L:int,J:float,R:int,PBC:bool):
    hop = np.array([J*(np.eye(L,k=1)+np.eye(L,k=-1)) for _ in range(R)],dtype=DTYPE)
    if PBC:
        hop[:,0,-1] = 1.
        hop[:,-1,0] = 1.
    return hop

def powerlaw_hopping(L:int,J:float,alpha:float,R:int,PBC:bool):
    def distance(i:int,j:int,L:int,PBC:bool):
        if PBC and abs(i-j) > L//2:
            return abs(abs(i-j)-L)
        return abs(i-j)
    
    J = np.zeros((R,L,L))
    for i in range(L):
        for j in range(i+1,L):
            d = distance(i, j, L, PBC)
            J[:, i, j] = np.random.normal(loc=0, scale=J/pow(d, alpha),size=R)
            J[:, j, i] = J[:, i, j]

    return J

# Power-law Random Banded Matrix
def PRBM(L:int,W:float,J:float,alpha:float,R:int=1,PBC:bool=True) -> np.ndarray:
    """ Spinless fermionic Hamiltonian with random uniform onsite potential and hopping terms which decay inversely to the distance between sites. In particular, the onsite potential at site :math:`i` is given by :math:`h_i = uniform(-W,W)` and the hopping term is given by :math:`J_{ij} = J/|i-j|^{\alpha}`.

    Args:
        L (int): Number of sites in the chain.
        W (float): Disorder strength. The onsite potential is chosen from a random uniform distribution in the interval :math:`[-W,W]`.
        J (float): Hopping amplitude
        alpha (float): _description_
        R (int, optional): Number of realizations. For any :math:`R>1`, then this function returns Hamiltonian matrices of shape `(R,L,L)`. Since this is a disordered model, it is useful to take averages over a certain number of realizations. Defaults to 1.
        PBC (bool, optional): Periodic boundary condition. Defaults to True.

    Returns:
        H2: The PRBM Hamiltonian as a matrix. If `R>1` the shape is `(R,L,L)` and `(L,L)` otherwise.
    """
    # h box distribution [-W,W]
    # J normal distribution var(J_ij) \prop J0/(\i-j|^alpha) where alpha=1 default
    h = disordered_potential(L,W,R)
    J = powerlaw_hopping(L,J,alpha,R,PBC)
    H2L = h + J

    return H2L

# Anderson Model
def tight_binding_model(L:int,W:float,J:float,R:int=1,PBC:bool=True)->np.ndarray:
    """ Spinless fermionic Hamiltonian with random uniform onsite potential 
    and hopping terms. In particular, the onsite potential at site :math:`i` is 
    given by :math:`h_i = uniform(-W,W)` and the hopping term is 
    given by :math:`J_{ij} = J` for :math:`|i-j| = 1`.

    Args:
        L (int): Number of sites in the chain.
        W (float): Disorder strength. The onsite potential is chosen from a random uniform distribution in the interval :math:`[-W,W]`.
        J (float): Hopping amplitude
        R (int, optional): Number of realizations. For any :math:`R>1`, then this function returns Hamiltonian matrices of shape `(R,L,L)`. Since this is a disordered model, it is useful to take averages over a certain number of realizations. Defaults to 1.
        PBC (bool, optional): Periodic boundary condition. Defaults to True.

    Returns:
        H2: The tight binding Hamiltonian as a matrix. If `R>1` the shape is `(R,L,L)` and `(L,L)` otherwise.
    """
    # h box distribution [-W,W]
    # J_{ij} = 1 if dist(i,j) = 1
    h = disordered_potential(L, W, R)
    hop = left_right_hopping(L,J,R,PBC)
    H2L = h + hop

    if R == 1: return H2L[0]
    return H2L


# Linear
def linear_potential_with_hopping(L:int,F:float,J:float,PBC:bool=True) -> np.ndarray:
    r""" Spinless fermionic Hamiltonian with linear potential and hopping terms.
    In particular, the onsite potential at site :math:`i` is given by 
    :math:`h_i = F×i` and the hopping term is given by :math:`J_{ij} = J` 
    for :math:`|i-j| = 1`.

    Args:
        L (int): Number of sites in the chain.
        F (float): _description_
        J (float): Hopping amplitude
        PBC (bool, optional): Periodic boundary condition. Defaults to True.

    Returns:
        H2: _description_
    """
    h = linear_potential(L,F,R)
    hop = left_right_hopping(L,J,1,PBC)[0]
    H2 = h +hop
    return H2
