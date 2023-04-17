import numpy as np
from sympy import Symbol, lambdify, sqrt, factorial
from scalar_product import scalarProduct


def PhiFunction(j):
    t = Symbol('t')
    return t ** j / factorial(j)


def getPhiBasisFunctions(p, T):
    """
    Given p, it should return a collection of basis functions of the form
    {(t^j)/j!} for j=0,...,p-1
    And orthonormalize them
    """
    non_normalized_basis = [PhiFunction(j) for j in range(p)]
    # Normalize basis functions
    orthonormal_basis = []
    for v in non_normalized_basis:
        u = v
        for e in orthonormal_basis:
            u = u - scalarProduct(v, e, p, T) / scalarProduct(e, e, p, T) * e
        orthonormal_basis.append(u / sqrt(scalarProduct(u, u, p, T)))
    return orthonormal_basis
