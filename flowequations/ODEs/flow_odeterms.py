import jax
import jax.numpy as jnp
from jax import lax
from jax import jit
from functools import partial
from .jax_comm import j_comm_vacform, j_comm_normal, j_comm
import numpy as np

"""
If you are reading this, you are probably really desperate to understand the code.
I don't feel like writing documentation here. 
I will say that masks are used to efficiently calculate the diagonal part of 
the Hamiltonian (tensor based operation, so it's faster on GPU too!). 
If you are still confused, please consider the following:
1. Refer to my thesis for a detailed explanation of the code.
2. If that doesn't work, try crying. It won't solve your problem, but I've 
   been told it helps?
3. Refer to my thesis again, but this time with a bottle of whiskey.

- zAsdf#1337
"""

# ------------- vacform ------------- #

def vacform_step(mask, t, H, *args):
    # quadratic part
    H0 = H*mask

    # calculate generator & RHS
    eta = j_comm_vacform(H0, H)
    sol = j_comm_vacform(eta, H)
    return sol

def vacform_step_dynamics(L,t, y, *args):
    H, = args

    yR = jax.lax.slice(y,(0,0,0), (1,L,L))
    yI = jax.lax.slice(y,(1,0,0), (2,L,L))

    HR = jnp.reshape(H, (L,L))

    solR = - j_comm_vacform(HR, yI)
    solI = j_comm_vacform(HR, yR)
    sol = jnp.concatenate((solR, solI))
    return jnp.reshape(sol,(2,L,L))

def ode_vacform(L, newL = -1, dynamics = False):
    if dynamics:
        if newL <= 0: raise Exception('ode_vacform: newL must be specified for dynamics')
        return jit(partial(vacform_step_dynamics, newL))
    
    # Mask allows omitting diagonal quartic part
    mask = np.diag(np.ones(L+np.sum(np.arange(L))))
    
    return jit(partial(vacform_step, jnp.array(mask)))


# ------------- tensor -------------- #

def tensor_step(L, mask2, mask4, t, H, *args):
    H2 = jnp.reshape(lax.slice(H, (0,), (L**2,)), (L, L))
    H4 = jnp.reshape(lax.slice(H, (L**2,), (L**2+L**4,)), (L, L, L, L))

    # quadratic part
    H0_2 = H2*mask2

    # quartic part
    H0_4 = H4*mask4

    # calculate generator & RHS
    eta2, eta4 = j_comm(H0_2, H0_4, H2, H4)
    sol2, sol4 = j_comm(eta2, eta4, H2, H4)
    return jnp.concatenate((jnp.ravel(sol2), jnp.ravel(sol4)))

def tensor_step_NI(L, mask2, t, H, *args):
    H2 = jnp.reshape(H, (L, L))

    # quadratic part
    H0_2 = H2*mask2

    # calculate generator & RHS
    eta2 = j_comm_vacform(H0_2, H2)
    sol2 = j_comm_vacform(eta2, H2)
    return jnp.ravel(sol2)

def tensor_step_dynamics(L, t, y, *args):
    H, = args
    H2 = jnp.reshape(lax.slice(H, (0,), (L**2,)), (L, L))
    H4 = jnp.reshape(lax.slice(H, (L**2,), (L**2+L**4,)), (L, L, L, L))

    y2R = jnp.reshape(lax.slice(y, (0,), (L**2,)), (L, L))
    y4R = jnp.reshape(lax.slice(y, (L**2,), (L**2+L**4,)), (L, L, L, L))
    y2I = jnp.reshape(lax.slice(y, (L**2+L**4,), (2*L**2+L**4,)), (L, L))
    y4I = jnp.reshape(lax.slice(y, (2*L**2+L**4,), (2*L**2+2*L**4,)), (L, L, L, L))

    # calculate generator & RHS
    sol2R, sol4R = j_comm(H2, H4, y2I, y4I)
    sol2I, sol4I = j_comm(H2, H4, y2R, y4R)
    sol2R = -sol2R
    sol4R = -sol4R
    return jnp.concatenate((jnp.ravel(sol2R), jnp.ravel(sol4R), jnp.ravel(sol2I), jnp.ravel(sol4I)))

def ode_tensor(L, non_interacting = False, dynamics = False):
    # Mask allows omitting diagonal quartic part
    if not dynamics:
        diag_indices = [(i, j, i, j) for i in range(L) for j in range(L)]
        diag_indices += [(i, j, j, i) for i in range(L) for j in range(L)]
        mask4 = np.zeros((L, L, L, L))
        for index in iter(diag_indices):
            mask4[index] = 1
        mask2 = np.diag(np.ones(L))
    

    # Make L & mask static
    if dynamics:
        return jit(partial(tensor_step_dynamics,L))
    if non_interacting:
        return jit(partial(tensor_step_NI,L,jnp.array(mask2)))
    return jit(partial(tensor_step, L, jnp.array(mask2), jnp.array(mask4)))

# ------------- normal -------------- #

def normal_step(L, mask2, mask4, n, t, H, *args):
    H2 = jnp.reshape(lax.slice(H, (0,), (L**2,)), (L, L))
    H4 = jnp.reshape(lax.slice(H, (L**2,), (L**2+L**4,)), (L, L, L, L))

    # quadratic part
    H0_2 = H2*mask2

    # quartic part
    H0_4 = H4*mask4

    # calculate generator & RHS
    eta2, eta4 = j_comm_normal(n,H0_2, H0_4, H2, H4)
    sol2, sol4 = j_comm_normal(n,eta2, eta4, H2, H4)
    return jnp.concatenate((jnp.ravel(sol2), jnp.ravel(sol4)))

def normal_step_dynamics(L, n, t, y, *args):
    H, = args
    H2 = jnp.reshape(lax.slice(H, (0,), (L**2,)), (L, L))
    H4 = jnp.reshape(lax.slice(H, (L**2,), (L**2+L**4,)), (L, L, L, L))

    y2R = jnp.reshape(lax.slice(y, (0,), (L**2,)), (L, L))
    y4R = jnp.reshape(lax.slice(y, (L**2,), (L**2+L**4,)), (L, L, L, L))
    y2I = jnp.reshape(lax.slice(y, (L**2+L**4,), (2*L**2+L**4,)), (L, L))
    y4I = jnp.reshape(lax.slice(y, (2*L**2+L**4,), (2*L**2+2*L**4,)), (L, L, L, L))

    # calculate generator & RHS
    sol2R, sol4R = j_comm_normal(n, H2, H4, y2I, y4I)
    sol2I, sol4I = j_comm_normal(n, H2, H4, y2R, y4R)
    sol2R = -sol2R
    sol4R = -sol4R
    return jnp.concatenate((jnp.ravel(sol2R), jnp.ravel(sol4R), jnp.ravel(sol2I), jnp.ravel(sol4I)))

def ode_normal(n, L, dynamics = False):
    # Mask allows omitting diagonal quartic part
    if not dynamics:
        diag_indices = [(i, j, i, j) for i in range(L) for j in range(L)]
        diag_indices += [(i, j, j, i) for i in range(L) for j in range(L)]
        mask4 = np.zeros((L, L, L, L))
        for index in iter(diag_indices):
            mask4[index] = 1
        mask2 = np.diag(np.ones(L))

    # Make L & mask static
    if dynamics:
        return jit(partial(normal_step_dynamics, L, n))
    return jit(partial(normal_step, L, jnp.array(mask2), jnp.array(mask4), n))

# ----------------------------------- #

