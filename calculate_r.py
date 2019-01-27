import numpy as np
from H_operator import Hi

def rVectorFromf( f, F, tSpan ):
    rVector=np.exp(-Hi(f,F,tSpan))

    return rVector