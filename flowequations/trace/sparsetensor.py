from collections import defaultdict
import numpy as np

"""
Source?
https://www.youtube.com/watch?v=r7l0Rq9E8MY
"""

class sparsetensor(object):
    def __init__(self, shape:tuple, **kwargs):
        opts = {
            'clean_after_add': False,
        }
        opts.update(kwargs)
        # I can't remember what clean_after_add was for. I think it was to 
        # optimize the trace calculation, but showed no benefit and never used it.
        self.clean_after_add = opts['clean_after_add'] 

        self.shape = shape
        self.data = defaultdict(float)

    def __getitem__(self, index:tuple):
        if len(index) != len(self.shape):
            raise ValueError("Wrong number of index")
        if not all(0 <= index[i] < self.shape[i] for i in range(len(self.shape))):
            raise IndexError("Index out of range")
        return self.data[index]

    def __setitem__(self, index:tuple, value:float):
        if len(index) != len(self.shape):
            raise ValueError("Wrong number of index")
        if not all(0 <= index[i] < self.shape[i] for i in range(len(self.shape))):
            raise IndexError("Index out of range")
        self.data[index] = value

    def __add__(self, other):
        if isinstance(other, tuple):
            index, value = other
            self.data[index] += value
            if self.clean_after_add:
                self.clean()
            return self
        
        if isinstance(other, SparseTensor):
            if self.shape != other.shape:
                raise ValueError("Tensors must have the same shape")
            if other.flatten_indexing:
                raise Error("Flattened -> non-flattened doesn't work")
            for index, value in other.data.items():
                self.data[index] += value
            if self.clean_after_add:
                self.clean()
            return self

        if isinstance(other, dict):
            for index, value in other.items():
                self.data[index] += value
            if self.clean_after_add:
                self.clean()
            return self
        
        if isinstance(other, list):
            for element in other:
                index, value = element
                self.data[index] += value
            if self.clean_after_add:
                self.clean()
            return self

        raise TypeError("Invalid operand type for +")
        
    def __mul__(self,A):
        if isinstance(A,np.ndarray):
            if A.shape == self.shape and A.dtype != object:
                f = lambda data: A[tuple(data[0][:])]*data[1]
            else:
                raise Exception("Something went wrong.")
        elif isinstance(A,tuple) and len(A) == 2:
            # TODO: implement for <class 'jaxlib.xla_extension.DeviceArray'>
            # if not all(isinstance(T,np.ndarray) for T in A):
            #     raise Exception("Something wrong.")
            B, C = A
            if B.shape+C.shape != self.shape:
                raise Exception("Yep. It's broken.")
            m = round((B.ndim+C.ndim)/2)
            u = round((B.ndim)/2)
            x = m+u

            f = lambda data: B[tuple(data[0][:u])+tuple(data[0][m:x])]*C[tuple(data[0][u:m])+tuple(data[0][x:])]*data[1]
        else:
            raise Exception("Suprise! An error.")

        return list(map(f,self.data.items()))


    def __repr__(self):
        return f"SparseTensor({self.shape}, {self.data})"

    def clean(self):
        self.data = {index: value for index, value in self.data.items() if value != 0}
        return self