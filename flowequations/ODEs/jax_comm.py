import jax.numpy as jnp
from jax import jit
from functools import partial

@jit
def j_comm22(A: jnp.ndarray,B: jnp.ndarray) -> jnp.ndarray:
    """ Commutator between two rank-2 tensors using jax.numpy.

    Args:
        A (jnp.ndarray): Rank-2 tensor
        B (jnp.ndarray): Rank-2 tensor

    Returns:
        jnp.ndarray: Commutator of A and B
    """
    return jnp.matmul(A, B) - jnp.matmul(B, A)

@jit
def j_comm42(A:jnp.ndarray,B:jnp.ndarray) -> jnp.ndarray:
    """ Commutator between rank-4 and rank-2 tensors using jax.numpy.

    Args:
        A (jnp.ndarray): Rank-4 tensor
        B (jnp.ndarray): Rank-2 tensor

    Returns:
        jnp.ndarray: Commutator of A and B
    """
    comm = jnp.einsum('qikl,jq->ijkl',A,B)
    comm = comm - jnp.einsum('qjkl,iq->ijkl',A,B)
    comm = comm + jnp.einsum('ijkq,ql->ijkl',A,B)
    comm = comm - jnp.einsum('ijlq,qk->ijkl',A,B)
    return comm

@jit
def j_comm24(A:jnp.ndarray,B:jnp.ndarray) -> jnp.ndarray:
    """ Commutator between rank-2 and rank-4 tensors using jax.numpy. This is exactly ``-1 * j_comm42(B,A)``.

    Args:
        A (jnp.ndarray): Rank-2 tensor
        B (jnp.ndarray): Rank-4 tensor

    Returns:
        jnp.ndarray: Commutator of A and B
    """
    return -1.*j_comm42(B,A)

@jit
def j_comm44(A:jnp.ndarray,B:jnp.ndarray) -> jnp.ndarray:
    """ Commutator between two rank-4 tensors using jax.numpy.

    Args:
        A (jnp.ndarray): Rank-4 tensor
        B (jnp.ndarray): Rank-4 tensor

    Returns:
        jnp.ndarray: Commutator of A and B
    """
    comm = jnp.tensordot(A,B,axes=([2,3],[0,1]))
    comm = comm - jnp.tensordot(B,A,axes=([2,3],[0,1]))
    return -2.*comm

@jit
def j_comm(A2:jnp.ndarray,A4:jnp.ndarray,B2:jnp.ndarray,B4:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ Commutator between two operators A & B, both containing rank-2 and rank-4 tensors. These contractions consider normal-ordering with respect to the vacuum. This function is intended to be combinded with `diffrax.ODETerm`.

    Args:
        A2 (jnp.ndarray): Rank-2 tensor of A
        A4 (jnp.ndarray): Rank-4 tensor of A
        B2 (jnp.ndarray): Rank-2 tensor of B
        B4 (jnp.ndarray): Rank-4 tensor of B

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: The commutator of tensors A and B
    """
    comm2 = j_comm22(A2,B2)
    comm4 = j_comm42(A4,B2)
    comm4 = comm4 + j_comm24(A2,B4)
    comm4 = comm4 + j_comm44(A4,B4)
    return comm2,comm4

### NORMAL COMM FUNCTIONS ###
@partial(jit, static_argnums=(0,))
def j_comm_normal(n:float,A2:jnp.ndarray,A4:jnp.ndarray,B2:jnp.ndarray,B4:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ Commutator between two operators A & B, both containing rank-2 and rank-4 tensors. These contractions consider normal-ordering with respect to the density matrix :math:`\\rho`, which sets the average particle number density :math:`n`. This function is intended to be combinded with `diffrax.ODETerm`.

    Args:
        n (float): The average particle number density set by the density matrix :math:`\\rho`. In particular, :math:`0 \\leq n \\leq 1`.
        A2 (jnp.ndarray): Rank-2 tensor of A
        A4 (jnp.ndarray): Rank-4 tensor of A
        B2 (jnp.ndarray): Rank-2 tensor of B
        B4 (jnp.ndarray): Rank-4 tensor of B

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: The commutator of tensors A and B
    """
    comm4 = j_comm42(A4,B2)
    comm4 = comm4 + j_comm24(A2,B4)
    comm44_1 = jnp.tensordot(A4, B4, axes=([2, 3], [0, 1]))
    comm44_2 = jnp.tensordot(B4, A4, axes=([2, 3], [0, 1]))
    comm4 = comm4 + 2*(2*n-1)*(comm44_1-comm44_2)

    comm2 = j_comm22(A2,B2)
    comm2_1 = jnp.tensordot(A4,B4, axes = ([1,2,3], [3,0,1]))
    comm2_2 = jnp.tensordot(B4,A4, axes = ([1,2,3], [3,0,1]))
    comm2 = comm2 + 8*n*(1-n)*(comm2_1-comm2_2)
    return comm2, comm4

### VACFORM COMM FUNCTIONS ###
@jit
def j_comm_vacform(A:jnp.ndarray,B:jnp.ndarray) -> jnp.ndarray:
    """ Commutator between two operators A & B using. These contractions consider normal-ordering with respect to the vacuum, where the tensors are in the vacuum-form. This function is intended to be combinded with `diffrax.ODETerm`.

    Args:
        A (jnp.ndarray): Rank-2 tensor in vacuum-form
        B (jnp.ndarray): Rank-2 tensor in vacuum-form

    Returns:
        jnp.ndarray: The commutator of tensors A and B in vacuum-form
    """
    comm = j_comm22(A,B)
    return comm