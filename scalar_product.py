from sympy import Symbol, diff, lambdify
from scipy.interpolate import UnivariateSpline
import numpy as np


def diffNtimes(myfunction, N):
    if N==0:
        return myfunction
    if isinstance(myfunction, UnivariateSpline):
        return myfunction.derivative(N)
    x = Symbol('tau')
    for _ in range(N):
        f = lambdify(x, diff(myfunction(x), x), "numpy")
    return f


def scalarProduct(function1,function2,p,tRange):
    firstSum=0
    for i in range(p):  #Will go up to p-1
        diffedFunction1=diffNtimes(function1,i)
        diffedFunction2 = diffNtimes(function2, i)
        firstSum+=diffedFunction1(0)*diffedFunction2(0)

    pDiffofFunction1=diffNtimes(function1,p)
    pDiffofFunction2 = diffNtimes(function2, p)
    pDiffofFunction1EvaluatedInTRange = np.array([pDiffofFunction1(t) for t in tRange])
    pDiffofFunction2EvaluatedInTRange = np.array([pDiffofFunction2(t) for t in tRange])

    integral=np.trapz(pDiffofFunction1EvaluatedInTRange*pDiffofFunction2EvaluatedInTRange,tRange)

    return firstSum+integral