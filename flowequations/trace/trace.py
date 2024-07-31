import numpy as np
import dill as pickle
import os

from .wicks import sparsetensors_fromwicks

"""
It just works. 
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

def second_trace_invariant_polynomial(H2:np.ndarray,H4:np.ndarray) -> np.ndarray:
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

def relative_trace_invariant_difference(H2i,H4i,H2f,H4f,n):
    inv_i = second_trace_invariant_polynomial(H2i,H4i)
    inv_f = second_trace_invariant_polynomial(H2f,H4f)
    x_i = np.sum([inv_i[s]*n**(s+1) for s in range(4)])
    x_f = np.sum([inv_f[s]*n**(s+1) for s in range(4)])
    if x_i == 0: return abs(x_f) # Bad? idk.
    return abs((x_f-x_i)/x_i)