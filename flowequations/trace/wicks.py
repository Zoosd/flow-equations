from .sparsetensor import sparsetensor
from itertools import permutations as perm

"""
See my thesis. Or just use the functions. Your call.
"""

coef_dict = {
    2: [-1, 1],
    3: [-1, 1, 1,-1,-1, 1],
    4: [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1]
}

def populate_sparse_tensor_bad(L, m, perms, part_coef,cleanup = True):
    st = sparsetensor(tuple([L for _ in range(2*m)]),clean_after_add= False)
    def f(ind): return o[ind]
    if m == 2:
        for i in range(L):
            for j in range(L):
                o = (i, j)
                for p, co in zip(perms, part_coef):
                    st + (o + tuple(map(f, p)),co)
    if m == 3:
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    o = (i, j, k)
                    for p, co in zip(perms, part_coef):
                        st + (o + tuple(map(f, p)), co)
    if m == 4:
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for l in range(L):
                        o = (i, j, k, l)
                        for p, co in zip(perms, part_coef):
                            st + (o + tuple(map(f, p)), co)
    if cleanup:
        return st.clean()
    return st

def R2_sparse_tensor(L):
    R2_st = sparsetensor((L,L),clean_after_add=False)
    for i in range(L):
        R2_st+((i,i),1)
    return R2_st

def sparsetensors_fromwicks(L,cleanup = True):
    st_list = []
    st_list.append(R2_sparse_tensor(L))
    for m in range(2,5):
        perms = list(perm(range(m), m))
        part_coef = coef_dict[m]
        st = populate_sparse_tensor_bad(L, m, perms, part_coef,cleanup = cleanup)
        st_list.append(st)
    return st_list
