import numpy as np
import dill as pickle
import os

from .wicks import sparsetensors_fromwicks

"""
Yes. The functions in this file have absurdly long names. I am aware of that. This is just done for clarity. Use imports to rename the functions once you are familiar with them.
"""

def product22(A, B):
    T21 = np.tensordot(A,B,axes=[0,1])
    arr = np.array([[T21,], [(-A,B),], [], []], dtype=object)
    return arr
    
def product24(A, B):
    T21 = np.tensordot(A,B,axes = [1,0])
    T31 = np.tensordot(A, B, axes=[1, 1])
    arr = np.array([[],[T21,-T31,],[(A,B),],[]],dtype=object)
    return arr
    
def product44(A, B):
    T31 = np.swapaxes(np.tensordot(A, B, axes=[2, 0]), 2, 3)
    T32 = np.swapaxes(np.tensordot(A, B, axes=[3, 0]), 2, 3)
    T41 = np.swapaxes(np.tensordot(A, B, axes=[2, 1]), 2, 3)
    T42 = np.swapaxes(np.tensordot(A, B, axes=[3, 1]), 2, 3)
    T4132 = np.einsum('abUUno->abno',T41)
    T3142 = np.einsum('abUUno->abno',T31)
    arr = np.array([[],[T4132,-T3142],[T31,-T32,-T41,T42],[(A,B)]],dtype=object)
    return arr

def second_trace_invariant_coefficients(H2:np.ndarray,H4:np.ndarray) -> np.ndarray:
    """ Calculates the second trace invariant polynomial. See my thesis for more information.

    Args:
        H2 (np.ndarray): Rank-2 of Hamiltonian in tensor form.
        H4 (np.ndarray): Rank-4 of Hamiltonian in tensor form.

    Raises: # TODO: Add exceptions
        Exception: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        coefficients (np.ndarray): An array of 4 coefficients corresponding to the polynomial :math:`c_0 + c_1 n + c_2 n^2 + c_3 n^3`.
    """
    L = len(H2)

    # Load pickled sparse tensor
    path = os.getcwd() + '/sparse_tensors/'
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            raise Exception(f"Could not create directory for path {path}")
    try:
        filepath = path + f'sparse_tensors_L{L}.cfg'
        if not os.path.exists(filepath):
            raise Exception()
        elif os.stat(filepath).st_size == 0:
            os.remove(filepath)
            raise Exception()
        with open(filepath,'rb') as reader:
            sparse_tensors = pickle.load(reader) # Security issue. Whoops.
    except:
        filepath = path + f'sparse_tensors_L{L}.cfg'
        sparse_tensors = sparsetensors_fromwicks(L)
        with open(filepath,'wb') as writer:
            pickle.dump(sparse_tensors, writer)
    
    prod_list = (product22(H2, H2), product44(H4, H4), 2*product24(H2, H4))
    val = np.zeros(4)
    # For each product case (22,24,44):
    for tensors_list in prod_list:
        # For each rank (2,4,6,8):
        for idx, tensors, sparse_tensor in zip(range(len(tensors_list)),tensors_list, sparse_tensors):
            if len(tensors) != 0:
                # For each non-empty tensor
                for tensor in tensors:
                    # Calculate the trace using the corresponding sparse tensor.
                    temp = sparse_tensor * tensor
                    val[idx] += np.sum(temp)
    return val 

def second_trace_invariant_polynomial(H2:np.ndarray, H4:np.ndarray, n:float) -> np.ndarray:
    """ Calculates the second trace invariant polynomial. See my thesis for more information.

    Args:
        H2 (np.ndarray): Rank-2 of Hamiltonian.
        H4 (np.ndarray): Rank-4 of Hamiltonian.
        n (float): The average particle number density set by the density matrix which was used for normal-ordering on the Hamiltonian during flow.

    Returns:
        np.ndarray: The second trace invariant polynomial evaluated at `n`. The shape is (4,).
    """
    coefficients = second_trace_invariant_coefficients(H2,H4)
    return np.asarray([coefficients[s]*n**(s+1) for s in range(4)])

def relative_difference_second_trace(H2i:np.ndarray, H4i:np.ndarray, H2f:np.ndarray, H4f:np.ndarray, n:float) -> float:
    """ Calculates the relative difference between the second trace invariant polynomials evaluated at the initial and final Hamiltonians.

    Args:
        H2i (np.ndarray): Rank-2 of Hamiltonian at the start of the flow.
        H4i (np.ndarray): Rank-4 of Hamiltonian at the start of the flow.
        H2f (np.ndarray): Rank-2 of Hamiltonian at the end of the flow.
        H4f (np.ndarray): Rank-4 of Hamiltonian at the end of the flow.
        n (float): The average particle number density set by the density matrix which was used for normal-ordering on the Hamiltonian during flow.

    Returns:
        float: The relative difference between the second trace invariant polynomials evaluated at the initial and final Hamiltonians.
    """
    trace_start = np.sum(second_trace_invariant_polynomial(H2i,H4i,n))
    trace_end = np.sum(second_trace_invariant_polynomial(H2f,H4f,n))
    if trace_start == 0: return abs(trace_end) # Bad? idk.
    return abs((trace_end-trace_start)/trace_start)

def relative_difference_second_trace_per_sector(L:int, trace_poly_i:np.ndarray, trace_poly_f:np.ndarray) -> np.ndarray:
    """ Calculates the relative difference of the second trace invariant between start and final Hamiltonians within all particle sectors available in the Hilbert space.
    For a spinless fermionic model in one-dimension, the Hilbert space `2^L` dimensional which can be divided into `L+1` subspaces. 
    Each subspace :math:`\\mathscr{H}^{(N)}` is :math:`L` choose :math:`N` dimensional containing :math:`N` particles.

    Args:
        L (int): The number of sites in the lattice.
        trace_poly_i (np.ndarray): The second trace invariant polynomial evaluated at the initial Hamiltonian.
        trace_poly_f (np.ndarray): The second trace invariant polynomial evaluated at the final Hamiltonian.

    Returns:
        np.ndarray: The relative difference of the second trace invariant between start and final Hamiltonians within all particle sectors available in the Hilbert space. The shape is (L+1).
    """
    
    def Cinvert(L): # See section 2.4.2 in my thesis. In particular, this is the matrix inverse of G in equation 2.107.
        fact = np.math.factorial
        C = np.zeros((L+1,L+1))
        for S in range(L+1):
            for N in range(S+1):
                C[S,N] = pow(-1,(S-N))*(fact(L-N))/(fact(S-N)*fact(L-S))
        return np.linalg.inv(C)
    
    def relative_difference(x,y): # Relative difference function
        if x == 0.: return abs(x-y)
        return abs(x-y)/x


    # Construct the inverse matrix of G in equation 2.107.
    C_inverted_matrix = Cinvert(L)

    # Allocate the trace per sector array and fill the first 4 sectors with the second trace invariant polynomial.
    inv_i, inv_f = np.zeros(L+1), np.zeros(L+1)
    inv_i[1:5], inv_f[1:5] = trace_poly_i, trace_poly_f

    # Calculate the second trace invariant within a particle sector.
    invPS_i, invPS_f = np.matmul(C_inverted_matrix,inv_i), np.matmul(C_inverted_matrix,inv_f)

    # Relative difference.
    trace_per_sector = np.array(list(map(relative_difference,invPS_i, invPS_f)))
                
    return trace_per_sector # Shape(L+1)