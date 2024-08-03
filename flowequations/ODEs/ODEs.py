import numpy as np
import math
import jax.numpy as jnp
import diffrax
from functools import partial
from .tensor_transformations import tensors_to_submatrices, submatrices_to_tensors, tensor_to_normal, normal_to_tensor
from .flow_odeterms import *

DTYPE = jnp.float64

# TODO:
# - Add better errors and exceptions
# - Ensure all these functions work as intended

def flow_equation_non_interacting(H2:np.ndarray, **kwargs) -> np.ndarray:
    """ The flow equation approach with Wegner's generator for a non-interacting fermionic model in one-dimension. Obtaining the diagonal entries are much more efficient through exact diagonalization. It is recommended not to use this function! This function makes use of `diffrax` and `jax` to efficiently solve the differential equation.

    Args:
        H2 (np.ndarray): The Hamiltonian as a matrix.

    Raises: # TODO: Better errors and exceptions
        Exception: _description_
        Exception: _description_

    Keyword Args:
        dt0 (float): The initial step size. Defaults to 0.001.
        solver (diffrax.Solver): The solver to be used. Defaults to `diffrax.Dopri5()`.
        max_steps (int): The maximum number of steps to be taken. Defaults to 1000000.


    Returns:
        H2 (np.ndarray): The "energy-diagonal" Hamiltonian.
    """

    # Flow options. Updated through kwargs
    FO = {
        'dt0': 0.001,
        'solver': diffrax.Dopri5(), 
        'max_steps': 1000000,
    }
    FO.update(kwargs)

    L = len(H2)

    # Set solver (default: Dopri5)
    solver = FO['solver']

    # adaptive stepsize controller
    stepsize_controller = diffrax.PIDController(rtol=1.e-7,         # Default
                                                atol=1.e-9,         # Default
                                                force_dtmin= False, # Default
                                                dtmin = FO['dt0'])  # Default
    
    steadystate = diffrax.SteadyStateEvent()
    diffeqsolve_kwargs = {
        'stepsize_controller': stepsize_controller,
        'max_steps': FO['max_steps'],
        'discrete_terminating_event': steadystate,
    }

    # Ensure Hermiticity & Symmetries
    if not custom_ishermitian(H2):
        raise Exception('flow_equation_non_interacting: Initial H2 not hermitian')

    # Set odestep (ODETerm)
    terms = diffrax.ODETerm(ode_tensor(L, non_interacting=True))
    y0 = jnp.ravel(jnp.array(H2, dtype = DTYPE))
    
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        FO['S'],
        FO['dt0'],
        y0,
        throw=False,
        **diffeqsolve_kwargs
    )

    (y1,) = sol.ys
    H2 = np.array(y1).reshape((L,L))

    if np.inf in H2 or np.nan in H2:
        raise Exception("flow_equation_non_interacting: infinty or NaN encountered in Hamiltonian")
    return H2

def flow_equation(H2:np.ndarray,H4:np.ndarray, method:str = 'vacform',**kwargs) -> tuple[np.ndarray, np.ndarray]:
    """ The flow equation approach with Wegner's generator for an interacting fermionic model in one-dimension. Normal-ordering was used as an appropriate truncation scheme. In particular, normal-ordering is done with respect to the density matrix :math:`\\rho` which sets the average particle number density :math:`n`. For more information, refer to `method` and `n`. This function makes use of `diffrax` and `jax` to efficiently solve the differential equation

    Args:
        H2 (np.ndarray): Rank-2 part of the Hamiltonian in tensor form
        H4 (np.ndarray): Rank-4 part of the Hamiltonian in tensor form
        method (str, optional): The method used for solving the flow equation. The options are `(tensor, normal, vacform)`. For vacuum ordering, use `tensor` (base implementation) or `vacform` (optimized implementation). For normal ordering, use `normal` and pass the average particle number density `n` in the keyword argument. Defaults to 'vacform'.

    Keyword Args:
        n (float): The average particle number density set by the density matrix :math:`\\rho`. In particular, :math:`0 \\leq n \\leq 1`. Required if `method` is `normal`. Defaults to 0.5 for half-filling normal-ordering.
        dt0 (float): The initial step size. Defaults to 0.001.
        solver (diffrax.Solver): The solver to be used. Defaults to `diffrax.Dopri5()`.
        max_steps (int): The maximum number of steps to be taken. Defaults to 1000000. This is just to ensure that this function will error if it takes too long (default number of steps would take extremely long!).

    Raises: TODO: Better errors and exceptions
        Exception: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        H2 (np.ndarray): The rank-2 part of the energy-diagonal Hamiltonian.
        H4 (np.ndarray): The rank-4 part of the energy-diagonal Hamiltonian.
    """
    # Flow options. Updated through kwargs
    FO = {
        'dt0': 0.001,
        'solver': diffrax.Dopri5(), 
        'max_steps': 1000000,
        'n': 0.5
    }
    FO.update(kwargs)

    # TODO: Checks
    L = len(H2)

    if not method in ['tensor', 'vacform', 'normal']:
        raise Exception('flow_equation: ODETerm not found. Options are tensor, vacform, normal')

    # Set solver (default: Dopri5)
    solver = FO['solver']


    # adaptive stepsize controller
    stepsize_controller = diffrax.PIDController(rtol=1.e-7,         # Default
                                                atol=1.e-9,         # Default
                                                force_dtmin= False, # Default
                                                dtmin = FO['dt0'])  # Default
    
    steadystate = diffrax.SteadyStateEvent(atol=1.e-7,rtol= 1.e-5)
    diffeqsolve_kwargs = {
        'stepsize_controller': stepsize_controller,
        'max_steps': FO['max_steps'],
        'discrete_terminating_event': steadystate,
    }


    # Ensure Hermiticity & Symmetries
    if not custom_ishermitian(H2):
        raise Exception('flow_equation: Initial H2 not hermitian')
    
    if not SymmetricQ(H4):
        raise Exception('flow_equation: Initial H4 not respecting symmetries')
    

    # Here we set the ODETerm and reshape the initial operator and Hamiltonian.
    # In particular, the ODETerms correspond to the following implementations:
    # ↳ tensor:     Original tensor based implementation
    # ↳ normal:     Non-trivial normal ordering (requires n to be passed!)
    # ↳ vacform:    Optimized vacuum ordering implementation
    
    # y0: Reshapes the initial Hamiltonian. For example
    # ↳ tensor:     y0.shape = (L**2 + L**4)
    # ↳ normal:     y0.shape = (L**2 + L**4)
    # ↳ vacform:    y0.shape = (L + (L 2), L + (L 2))
    # Note: (x y) means x choose y.
    # Note: Hamiltonian also gets reshaped into these forms.
    # Note: y is a jnp.array. This is used for jax compatibility (works for both
    #       CPU and GPU!).
    if method == 'tensor': # TODO odefunc rename
        odefunc = ode_tensor
        H2 = jnp.ravel(jnp.array(H2,dtype=DTYPE))
        H4 = jnp.ravel(jnp.array(H4,dtype=DTYPE))
        y0 = jnp.concatenate((H2,H4))
    elif method == 'vacform':
        odefunc = ode_vacform
        Hsub1, Hsub2 = tensors_to_submatrices(H2,H4)
        y0 = np.zeros((L+len(Hsub2),L+len(Hsub2)))
        y0[:L,:L] = Hsub1
        y0[L:,L:] = Hsub2
        y0 = jnp.array(y0,dtype=DTYPE)
    elif method == 'normal':
        if FO['n'] is None:
            raise Exception('flow_equation: normal requires parameter \'n\' (float) to be passed')
        if FO['n'] < 0 or FO['n'] > 1:
            raise Exception('flow_equation: normal requires parameter \'n\' (float) to be in the range [0,1]')
        odefunc = partial(ode_normal, FO['n'])
        _, H2n, H4n = tensor_to_normal(FO['n'], H2, H4)
        H2n = jnp.ravel(jnp.array(H2n,dtype=DTYPE))
        H4n = jnp.ravel(jnp.array(H4n,dtype=DTYPE))
        y0 = jnp.concatenate((H2n,H4n))
    
    # Use diffrax.diffeqsolve to solve the ODE. The solution is stored in sol.
    # See diffrax documentation for more information on what sol contains.
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(odefunc(L)),
        solver,
        0.,
        jnp.inf,
        FO['dt0'],
        y0,
        throw=False,
        **diffeqsolve_kwargs
    )

    (y1,) = sol.ys

    # Extract the flowed Hamiltonian and transform back into tensor form.
    y1 = np.array(y1)
    if np.isnan(y1).any():
        raise Exception('flow_equations: NaN encountered after flow')
    if method == 'vacform':
        H2, H4 = submatrices_to_tensors(np.array(y1[:L,:L]),np.array(y1[L:,L:]))
    elif method == 'normal':
        _, H2, H4 = normal_to_tensor(FO['n'], np.array(y1[:L**2]).reshape((L,L)), np.array(y1[L**2:]).reshape((L,L,L,L)))
    else:
        H2 = np.array(y1[:L**2]).reshape((L,L))
        H4 = np.array(y1[L**2:]).reshape((L,L,L,L))

    # Ensure Hermiticity & Symmetries
    if not custom_ishermitian(np.around(H2,10)):
        raise Exception('flow_equations: H2 not hermitian after flow equations')
    
    if not SymmetricQ(np.around(H4,10)):
        raise Exception('flow_equations: H4 not respecting symmetries after flow equations')

    return H2, H4

def truncated_equations_of_motion(H2:np.ndarray, H4:np.ndarray, op2:np.ndarray, op4:np.ndarray, interactions:bool, verbose:bool = False, method:str = 'vacform', **kwargs) -> tuple[np.ndarray, np.ndarray,np.ndarray]:
    """ Time-evolve the operator using Heisenberg's equations of motion. Here we impose the approximation scheme which only contains the terms in the Hamiltonian and operator up to the quartic order. This function makes use of `diffrax` and `jax` to efficiently solve the differential equation

    Args:
        H2 (np.ndarray): The quadratic part of the Hamiltonian (rank-2 tensor)
        H4 (np.ndarray): The quartic part of the Hamiltonian (rank-4 tensor)
        op2 (np.ndarray): The quadratic part of the operator at :math:`t=0` (rank-2 tensor)
        op4 (np.ndarray): The quartic part of the operator at :math:`t=0` (rank-4 tensor)
        interactions (bool): If False, then we consider the non-interacting case (only quadratic part).
        verbose (bool, optional): If True, then we return the full operator at each time step. Otherwise, we only return the final operator. Defaults to False.
        method (str, optional): The method used for solving the flow equation. The options are `(tensor, normal, vacform)`. For vacuum ordering, use `tensor` (base implementation) or `vacform` (optimized implementation). For normal ordering, use `normal` and pass the average particle number density `n` in the keyword argument. Defaults to 'vacform'.

    Keyword Args:
        n (float): The average particle number density set by the density matrix :math:`\\rho`. In particular, :math:`0 \\leq n \\leq 1`. Required if `method` is `normal`. Defaults to 0.5 for half-filling normal-ordering.
        dt0 (float): The initial step size. Defaults to 0.001.
        S (float): The final time. Defaults to 150.
        steptype (str): The type of step size. Options are `(log, linear)`. Defaults to 'log'.
        steps (int): The number of steps to be saved. Defaults to 5000.
        solver (diffrax.Solver): The solver to be used. Defaults to `diffrax.Dopri5()`.

    TODO: Better errors and exceptions
    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        op2 (np.ndarray): The time-evolved rank-2 operator in tensor form. Contains the operator at each time step and is of shape `(steps,L,L)`. If verbose is False, then `steps = 1`
        op4 (np.ndarray): The time-evolved rank-4 operator in tensor form. Contains the operator at each time step and is of shape `(steps,L,L,L,L)`. If verbose is False, then `steps = 1`
        ts (np.ndarray): The time steps at which the operator is saved.
    """
    
    # Flow options. Updated through kwargs.
    FO = {
        'dt0': 0.001,                   # Initial step size  
        'S': 150.,                      # Final time
        'steptype': 'log',              # Step type (log, linear)
        'steps': 5000,                  # Number of steps to be saved
        'solver': diffrax.Dopri5(),     # Solver to be used (default: Dopri5, see diffrax for more options)
        'n': 0.5,                       # Average particle number density (for non-trivial normal-ordering)
    }
    FO.update(kwargs)

    # TODO: Checks
    L = len(H2)

    # Set initial & final time
    steps = FO['steps']
    if FO['steps'] > 2:
        if FO['steptype'] == 'log':         # Logarithmic step size
            base = 10
            ph1 = FO['dt0']
            ph2 = FO['S']
            ts = np.logspace(math.log(ph1,base), math.log(ph2, base), num = FO['steps']-1, base = base, endpoint= True)
            ts = np.insert(ts,0, 0)
        elif FO['steptype'] == 'linear':    # Linear step size
            ts = np.linspace(0., FO['S'], num = FO['steps'], endpoint= True)
        else: raise Exception('bad steptype')
    elif FO['steps'] == 2:
        ts = np.array([0., FO['S']])
    else:
        raise Exception('bad steptype')
    
    jnpts = jnp.array(ts)

    # Set solver (default: Dopri5)
    solver = FO['solver']

    # adaptive stepsize controller
    stepsize_controller = diffrax.PIDController(rtol=1.e-7,         # Default
                                                atol=1.e-9,         # Default
                                                force_dtmin= False, # Default
                                                dtmin = 1.e-5)      # Default

    # Kwargs for diffeqsolver (Written like this to easily add steadystate termination)
    diffeqsolve_kwargs = {
        'saveat': diffrax.SaveAt(ts=jnpts),             # Save at specific times
        'stepsize_controller': stepsize_controller,     # Adaptive stepsize controller
        'max_steps': None,                              # The number of steps required to reach the final time is unlimited
    }

    # Ensure Hermiticity & Symmetries
    if not custom_ishermitian(H2):
        raise Exception('flow_dynamics: Initial H2 not hermitian')
    
    if interactions and not SymmetricQ(H4):
        raise Exception('flow_dynamics: Initial H4 not respecting symmetries')
    
    # Here we set the ODETerm and reshape the initial operator and Hamiltonian.
    # In particular, the ODETerms correspond to the following implementations:
    # ↳ tensor:     Original tensor based implementation
    # ↳ normal:     Non-trivial normal ordering (requires n to be passed!)
    # ↳ vacform:    Optimized vacuum ordering implementation
    
    # y: Takes the initial Operator (to be time-evolved) and reshapes it into
    #    corresponding shapes. Contains real and imaginary parts. For example
    # ↳ tensor:     y.shape = (2, L**2 + L**4)
    # ↳ normal:     y.shape = (2, L**2 + L**4)
    # ↳ vacform:    y.shape = (2, L + (L 2), L + (L 2))
    # Note: (x y) means x choose y.
    # Note: Hamiltonian also gets reshaped into these forms.
    # Note: y is a jnp.array. This is used for jax compatibility (works for both
    #       CPU and GPU!).

    # If interactions are turned off, used vacform ODETerm (no quartic part)
    if not interactions: 
        terms = diffrax.ODETerm(ode_vacform(L, dynamics= True, newL = L))
        H = jnp.array(H2, dtype = DTYPE)
        y = np.zeros((2,)+op2.shape, dtype = DTYPE)
        y[0] = op2
        y = jnp.array(y, dtype = DTYPE)
    elif method == 'tensor':
        terms = diffrax.ODETerm(ode_tensor(L, dynamics= True))
        H2 = jnp.ravel(jnp.array(H2,dtype=DTYPE))
        H4 = jnp.ravel(jnp.array(H4,dtype=DTYPE))
        H = jnp.concatenate((H2,H4), dtype=DTYPE)
        y2R = jnp.ravel(jnp.array(op2,dtype=DTYPE))
        y4R = jnp.ravel(jnp.array(op4,dtype=DTYPE))
        y = jnp.concatenate((y2R,y4R, jnp.zeros(L**2 + L**4)), dtype=DTYPE)
    elif method == 'matform':
        Hsub1, Hsub2 = tensors_to_submatrices(H2,H4)
        Ysub1, Ysub2 = tensors_to_submatrices(op2,op4)
        newL = L+len(Hsub2)
        terms = diffrax.ODETerm(ode_vacform(L,dynamics=True, newL = newL))
        y = np.zeros((newL,newL))
        H = np.zeros((newL,newL))
        H[:L,:L] = Hsub1
        y[:L,:L] = Ysub1
        H[L:,L:] = Hsub2
        y[L:,L:] = Ysub2
        H = jnp.array(H,dtype=DTYPE)
        y = jnp.concatenate((jnp.ravel(y), jnp.ravel(jnp.zeros((newL,newL)))))
        y = jnp.reshape(y, (2,newL,newL))
    elif method == 'normal':
        if FO['n'] is None:
            raise Exception('flow_dynamics: normal requires parameter \'n\' (float) to be passed')
        if FO['n'] < 0 or FO['n'] > 1:
            raise Exception('flow_dynamics: normal requires parameter \'n\' (float) to be in the range [0,1]')
        terms = diffrax.ODETerm(ode_normal(FO['n'],L,dynamics=True))
        _, H2n, H4n = tensor_to_normal(FO['n'], H2, H4)
        _, y2nR, y4nR = tensor_to_normal(FO['n'], op2, op4)
        H2n = jnp.ravel(jnp.array(H2n,dtype=DTYPE))
        H4n = jnp.ravel(jnp.array(H4n,dtype=DTYPE))
        y2nR = jnp.ravel(jnp.array(y2nR,dtype=DTYPE))
        y4nR = jnp.ravel(jnp.array(y4nR,dtype=DTYPE))
        H = jnp.concatenate((H2n,H4n),dtype=DTYPE)
        y = jnp.concatenate((y2nR,y4nR, jnp.zeros(L**2 + L**4)),dtype=DTYPE)
    else:
        raise Exception('flow_dynamics: ODETerm not found.')
    
    # Use diffrax.diffeqsolve to solve the ODE. The solution is stored in sol.
    # See diffrax documentation for more information on what sol contains.
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        ts[0],
        ts[-1],
        None,
        y,
        args = H,
        throw=True,
        **diffeqsolve_kwargs,
    )

    # Extract the real part of the solution
    def get_real(arr):
        shape = jnp.shape(arr)
        arr = jnp.ravel(arr)
        mid_point = jnp.size(arr)//2
        arr1 = jax.lax.slice(arr, (0,), (mid_point,))
        newshape = shape[1:]
        if newshape == ():
            newshape = (shape[0]//2,)
        return arr1.reshape(newshape)
    
    # Convert the operator to np.ndarray to perform some checks efficient on the CPU.
    # Jitted functions are only efficient when they are run multiple times.
    ops = np.array(jax.lax.map(get_real,sol.ys))

    if np.isnan(ops).any() or np.isinf(ops).any():
        raise Exception('flow_dynamics: NaN or Inf encountered in operator after flow')

    # Reshape the operator to the correct shape
    # If verbose is true, we return the full operator at each time step.
    # Otherwise, we only return the final operator.    
    if verbose:
        if not interactions:
            ops2 = ops.reshape((steps,L,L))
            ops4 = None
        elif method == 'matform':
            ops2, ops4 = np.array(list(map(submatrices_to_tensors,ops[:,:L,:L],ops[:,L:,L:])))
        elif method == 'normal':
            f = partial(normal_to_tensor,FO['n'])
            ops2,ops4 = np.array(list(map(f,ops[:,:L**2].reshape((steps, L,L)), ops[:,L**2:].reshape((steps,L,L,L,L)))))
        elif method == 'tensor':
            ops2 = ops[:,:L**2].reshape((steps,L,L))
            ops4 = ops[:,L**2:].reshape((steps,L,L,L,L))
    else: # Return only the final operator, they are in a list of length 1 to keep the same format as verbose
        if not interactions:
            ops2 = np.array([ops[-1].reshape((L,L)),])
            ops4 = None
        elif method == 'matform':
            ops2, ops4 = submatrices_to_tensors(ops[-1,:L,:L],ops[-1,L:,L:])
            ops2 = np.array([ops2,])
            ops4 = np.array([ops4,])
        elif method == 'normal':
            f = partial(normal_to_tensor,FO['n'])
            ops2, ops4 = f(ops[-1,:L**2].reshape((L,L)), ops[-1,L**2:].reshape((L,L,L,L)))
            ops2 = np.array([ops2,])
            ops4 = np.array([ops4,])
        elif method == 'tensor':
            ops2 = np.array([ops[-1,:L**2].reshape((L,L)),])
            ops4 = np.array([ops[-1,L**2:].reshape((L,L,L,L)),])
    
    return ops2, ops4, np.array(ts)


def SymmetricQ(A4:np.ndarray,cutoff:float=1.e-7) -> bool:
    """ Checks symmetry properties of rank-4 tensors.

    Args:
        A4 (np.ndarray): Rank-4 tensor
        cutoff (float, optional): tolerance. Defaults to 1.e-7.

    Returns:
        bool: Returns True if symmetries are being respected, False otherwise.
    """
    L = len(A4)
    ret = True
    B4_1 = -np.swapaxes(A4, 0, 1)
    B4_2 = -np.swapaxes(A4, 2, 3)
    B4_3 = np.swapaxes(np.swapaxes(A4, 0, 1), 2, 3)
    
    B4_list = [B4_1,B4_2,B4_3]
    B4_names = ['-B_jikl','-B_ijlk','B_jilk']
    
    for B4,name in zip(B4_list,B4_names):
        diff = np.array(abs(A4-B4)).reshape((L,L,L,L))
        if cutoff != None:
            diff = np.where(diff > cutoff, diff, 0.)
        if len(np.argwhere(diff)) != 0:
            print(f"SymmetricQ: Rank-4 not symmetric ({name})")
            print("Printing all indices corresponding to nonzero values:")
            for ind in np.argwhere(diff):
                print(f"{ind} -> {diff[tuple(ind)]}")
            ret = False
    return ret

def custom_ishermitian(A2:np.ndarray, cutoff:float = 1.e-7) -> bool:
    """ Custom hermiticity check for rank-2 tensors.

    Args:
        A2 (np.ndarray): Rank-2 tensor
        cutoff (float, optional): tolerance. Defaults to 1.e-7.

    Returns:
        bool: Returns True if the tensor is hermitian, False otherwise.
    """
    A2t = np.transpose(A2)
    diff = abs(A2t - A2)
    temp = np.count_nonzero(np.where(diff>cutoff,diff,0.))
    return temp==0