import numpy as np

class indexdict(dict):
    """ Dictionary with missing key exception """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __missing__(self,key):
        raise Exception(f"Key missing. ")
    
def build_index_dictionary(L:int) -> indexdict:
    """ Construct index dictionary for a system of :math:`L` sites

    Args:
        L (int): Number of sites in the system

    Returns:
        indexdict: Dictionary with keys :math:`(i,j)` and values corresponding to the index of the :math:`(i,j)`'th element in a flattened matrix
    """
    index_dictionary = indexdict()
    c = 0
    for i in range(L):
        for j in range(i):
            index_dictionary.update({(i,j): c})
            c += 1
    return index_dictionary

def tensors_to_submatrices(H2:np.ndarray,H4:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """ Convert Hamiltonian to submatrices used for vacuum-form.

    Args:
        H2 (np.ndarray): Rank-2 part of Hamiltonian
        H4 (np.ndarray): Rank-4 part of Hamiltonian

    Returns:
        Hsub1 (np.ndarray): First submatrix of Hamiltonian containing the quadratic part.
        Hsub2 (np.ndarray): Second submatrix of Hamiltonian containing the interactions. The complete vacuum-form Hamiltonian is given in the block-diagonal form as :math:`H = [[Hsub1, 0], [0, Hsub2]]`.
    """
    Hsub1 = H2
    L = len(H2)
    Lsub2 = np.sum(np.arange(L))
    Hsub2 = np.zeros((Lsub2,Lsub2))
    ind = build_index_dictionary(L)
    for i in range(L):
        for j in range(i):
            for ip in range(L):
                for jp in range(ip):
                    Hsub2[ind[(i,j)],ind[(ip,jp)]] -= 4*H4[i,j,ip,jp]
                    if j == jp:
                        Hsub2[ind[(i, j)], ind[(ip, jp)]] += H2[i,ip]
                    if i == jp:
                        Hsub2[ind[(i, j)], ind[(ip, jp)]] -= H2[j, ip]
                    if j == ip:
                        Hsub2[ind[(i, j)], ind[(ip, jp)]] -= H2[i, jp]
                    if i == ip:
                        Hsub2[ind[(i, j)], ind[(ip, jp)]] += H2[j, jp]

    return Hsub1, Hsub2


def submatrices_to_tensors(Hsub1:np.ndarray, Hsub2:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """ Convert Hamiltonian submatrices to tensors. This is the inverse operation of `tensors_to_submatrices`.

    Args:
        Hsub1 (np.ndarray): First submatrix of Hamiltonian containing the quadratic part.
        Hsub2 (np.ndarray): Second submatrix of Hamiltonian containing the interactions.

    Returns:
        H2 (np.ndarray): Rank-2 part of Hamiltonian.
        H4 (np.ndarray): Rank-4 part of Hamiltonian.
    """
    H2 = Hsub1
    L = len(H2)
    H4 = np.zeros((L, L, L, L))
    ind = build_index_dictionary(L)
    for i in range(L):
        for j in range(i):
            for ip in range(L):
                for jp in range(ip):
                    temp = 0
                    if j == jp:
                        temp += H2[i, ip]
                    if i == jp:
                        temp -= H2[j, ip]
                    if j == ip:
                        temp -= H2[i, jp]
                    if i == ip:
                        temp += H2[j, jp]

                    temp = (Hsub2[ind[(i, j)], ind[(ip, jp)]] - temp)/4
                    H4[i, j, ip, jp] -= temp
                    H4[j, i, ip, jp] += temp
                    H4[i, j, jp, ip] += temp
                    H4[j, i, jp, ip] -= temp

    return H2,H4

def tensor_to_normal(n:float, H2:np.ndarray,H4:np.ndarray) -> tuple[float,np.ndarray,np.ndarray]:
    """ Convert Hamiltonian in tensor form to normal-ordered form. The normal-ordering is done with respect to density matrix :math:`\\rho` which sets the average particle number density :math:`\\rho`, which sets the average particle number density :math:`n`.

    Args:
        n (float): The average particle number density set by the density matrix :math:`\\rho`. In particular, :math:`0 <= n <= 1`.
        H2 (np.ndarray): Rank-2 part of Hamiltonian in tensor form.
        H4 (np.ndarray): Rank-4 part of Hamiltonian in tensor form.

    Returns:
        H0n (float): The constant part of the normal-ordered Hamiltonian which may be ignored.
        H2n (np.ndarray): Rank-2 part of the normal-ordered Hamiltonian.
        H4n (np.ndarray): Rank-4 part of the normal-ordered Hamiltonian.
    """
    # Rank 2 contribution
    H2n = np.copy(H2)
    H0n = n*np.sum(np.diag(H2n))

    # Rank 4 contribution
    H4_contribution = np.einsum('ikkj->ij',H4)
    H2n += 4*n*H4_contribution
    H0n += 2*n*n*np.sum(np.diag(H4_contribution))

    return H0n, H2n, np.copy(H4)

def normal_to_tensor(n:float, H2:np.ndarray,H4:np.ndarray) -> tuple[float,np.ndarray,np.ndarray]:
    """ Convert the normal-ordered Hamiltonian back to tensor form. The normal-ordering is done with respect to density matrix :math:`\\rho` which sets the average particle number density :math:`\\rho`, which sets the average particle number density :math:`n`. This is the inverse operation of `tensor_to_normal`, which should use the same value of :math:`n`.

    Args:
        n (float): The average particle number density set by the density matrix :math:`\\rho`. In particular, :math:`0 <= n <= 1`.
        H2 (np.ndarray): Rank-2 part of the normal-ordered Hamiltonian.
        H4 (np.ndarray): Rank-4 part of the normal-ordered Hamiltonian.

    Returns:
        H0n (float): The constant part of the normal-ordered Hamiltonian which may be ignored.
        H2n (np.ndarray): Rank-2 part of the Hamiltonian in tensor form.
        H4n (np.ndarray): Rank-4 part of the Hamiltonian in tensor form.
    """
    # Rank 2 contribution
    H2n = np.copy(H2)
    H0n = -n*np.sum(np.diag(H2n))

    # Rank 4 contribution
    H4_contribution = np.einsum('ikkj->ij',H4)
    H2n -= 4*n*H4_contribution
    H0n += 2*n*n*np.sum(np.diag(H4_contribution))

    return H0n, H2n, np.copy(H4)
    

