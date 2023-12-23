import numpy as np


def numerical_gradient(fun, x, eps=1e-4):
    """Compute derivatives of a given function fun numerically.
    
    Args:
      fun: A python function fun(x) which accepts a vector argument (one-dimensional numpy array)
           and returns a vector output (one-dimensional numpy array).
      x:   An input vector for which the numerical gradient should be computed.
      eps: A scalar which defines the magnitude of perturbations applied to the inputs.

    Returns:
      gnum: A two-dimensional array in which an element in row i and column j is the partial derivative of the
            i-th output of function fun wrt j-th input of function fun (computed numerically).
    """
    assert x.ndim <= 1, "Only vector inputs are supported"
    e = np.zeros_like(x)
    f = fun(x)
    assert f.ndim <= 1, "Only vector outputs are supported"
    gnum = np.zeros((f.size, x.size))
    for i in range(len(x)):
        e[:] = 0
        e[i] = 1
        f1, f2 = fun(x + e*eps), fun(x - e * eps)
        gnum[:, i] = (f1 - f2) / (2 * eps)
    return gnum