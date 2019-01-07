import numpy as np
from H_operator import Hi

def rVectorFromf( f, F, tSpan ):
    rVector=np.concatenate([[0],np.array([ Hi(f,F,tSpan[:i]) for i in range(1,len(tSpan)) ])])/tSpan
    return rVector