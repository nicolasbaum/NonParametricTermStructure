############################################################
# scalar_product.py
#
# Provides:
#   scalarProduct(f1, f2, p, tRange)
#
# Which implements the inner product:
#   <f, g> = sum_{s=0 to p-1} [ f^(s)(0) * g^(s)(0) ]  +  ∫ [f^(p)(t) * g^(p)(t)] dt
#
# In practice we do the discrete "boundary term" by the array's first element 
# after s times finite-differencing. Then do a numeric trapz for the p-th difference.
# 
#   1) If f1/f2 is a callable (e.g. InterpolatedUnivariateSpline), we evaluate it at tRange => array.
#   2) Then we call repeatedDiff(...) up to p times.
#   3) sum boundary terms for s < p, then integrate the p-th difference product.
#
# NOTE: If your model truly needs the derivative in the "splines sense" (like .derivative()), 
# or you want actual partial derivatives, you can adapt repeatedDiff(...) 
# to call f1.derivative(...) etc. 
############################################################

import numpy as np

def to_array(possibleFunc, tRange):
    """
    Convert 'possibleFunc' into a numpy array over tRange.
      - If it's already an ndarray, return it directly.
      - If it's callable (e.g. a spline or lambda), call it with tRange => array.
      - Otherwise, raise TypeError.
    """
    if isinstance(possibleFunc, np.ndarray):
        return possibleFunc

    if callable(possibleFunc):
        # Evaluate the function at tRange => should get numeric array
        arr = possibleFunc(tRange)
        # If it's list-like, convert
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=float)
        return arr

    raise TypeError(
        f"to_array: cannot handle object of type {type(possibleFunc)}. "
        "Expected a numeric ndarray or a callable (spline/lambda)."
    )

def repeatedDiff(arr, n):
    """
    Perform n-fold discrete difference on a copy of 'arr'.
    Because the code needs same-length arrays for 
    boundary indexing, we'll do a 'leading zero' approach:
      repeatedDiff([f0,f1,f2,...], 1) => [0, (f1-f0), (f2-f1), ...]
    So it keeps 'arr' the same size.
    
    If n=0, just return arr as is.
    """
    out = arr.copy()
    for _ in range(n):
        dif = np.diff(out)
        # prepend a zero for shape consistency:
        dif = np.concatenate(([0.0], dif))
        out = dif
    return out

def scalarProduct(f1, f2, p, tRange):
    """
    The "spline inner product" often used in the Lapshin–Kaushanskiy approach:
    
      <f, g> = sum_{s=0 to p-1} [ f^(s)(0) * g^(s)(0) ]  +  ∫ [ f^(p)(t) * g^(p)(t) ] dt
      
    In a discrete sense, we interpret "f^(s)(0)" as 
    the first element of repeatedDiff(...) of length-len(tRange).
    
    Steps:
      1) Convert f1/f2 => arrays over tRange => arr1, arr2
      2) For s in 0..p-1:
           df1_s = repeatedDiff(arr1, s)
           df2_s = repeatedDiff(arr2, s)
           boundarySum += df1_s[0] * df2_s[0]
      3) df1_p = repeatedDiff(arr1, p)
         df2_p = repeatedDiff(arr2, p)
         integralPart = trapz( df1_p * df2_p, x=tRange )
      4) return boundarySum + integralPart
    """
    arr1 = to_array(f1, tRange)
    arr2 = to_array(f2, tRange)

    # Basic shape check
    if arr1.shape != arr2.shape:
        raise ValueError(f"Arrays differ in shape: {arr1.shape} vs {arr2.shape}")

    # sum_{s=0}^{p-1} f^(s)(0)*g^(s)(0)
    boundarySum = 0.0
    for s in range(p):
        df1_s = repeatedDiff(arr1, s)
        df2_s = repeatedDiff(arr2, s)
        # "value at zero" ~ first element
        boundarySum += df1_s[0] * df2_s[0]

    # the integral part with p-th diff
    df1_p = repeatedDiff(arr1, p)
    df2_p = repeatedDiff(arr2, p)

    # numeric trapezoid integral
    integralPart = np.trapz(df1_p * df2_p, x=tRange)

    return boundarySum + integralPart

def getNormOfFunction(f, tRange, p):
    """
    Norm from the same inner product:
      ||f|| = sqrt( <f, f> ).
    """
    val = scalarProduct(f, f, p, tRange)
    return np.sqrt(val) if val>0 else 0.0