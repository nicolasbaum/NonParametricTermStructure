from sympy import Symbol, diff, lambdify
import numpy as np

def diffNtimes(myfunction, N):
    x = Symbol('tau')
    for _ in range(N):
        f = lambdify(x, diff(myfunction(x), x), "numpy")
    return f

def scalarProduct(function1,function2,p,tRange):
    firstSum=0
    for i in range(p-1):
        diffedFunction1=diffNtimes(function1,i)
        diffedFunction2 = diffNtimes(function2, i)
        firstSum+=diffedFunction1(0)*diffedFunction2(0)

    pDiffofFunction1=diffNtimes(function1,p)
    pDiffofFunction2 = diffNtimes(function2, p)
    integral=np.trapz(pDiffofFunction1(tRange)*pDiffofFunction2(tRange),tRange)

    return firstSum+integral