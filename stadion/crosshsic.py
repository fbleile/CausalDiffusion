# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from jax import numpy as jnp
import jax.lax as lax
import jax

from functools import partial
import scipy.stats as stats

CROSS_HSIC_TH = stats.norm.ppf(1-0.05) # alpha = 0.05

@jax.jit
def RBFkernel(x, y=None, bw=5.0):
    """Compute the RBF kernel between batches of x and y."""
    
    # Default `y` is set to `x` if None is provided.
    y = x if y is None else y
    
    # Use `vmap` to calculate the kernel for all pairs of x and y
    def kernel_pairwise(x_i, y_j):
        return jnp.exp(-(x_i - y_j) ** 2 / (2 * bw * bw))  # RBF kernel

    # Vectorize the kernel computation across batches
    vmap_kernel = jax.vmap(jax.vmap(kernel_pairwise, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=0)

    return vmap_kernel(x, y)  # Compute the kernel matrix

@jax.jit
def get_T1(K, L, n=None):
    if n is None:
        N = len(K)
        assert N % 2 == 0
        n = N // 2
    # Pointwise product
    M = K * L
    # Sum the last n columns of the 2D array M
    Mu = jnp.sum(M[:, n:], axis=1)
    # Sum the first n terms of the 1D array Mu
    T1 = (1 / (n * n)) * jnp.sum(Mu[:n])
    return T1

@jax.jit
def get_T2(K, L, n=None):
    if n is None:
        N = len(K)
        assert N % 2 == 0
        n = N // 2
    # Multiply the two kernel matrices
    KL = jnp.matmul(K, L)
    # Sum the last n columns of KL
    term10 = jnp.sum(KL[:, n:], axis=1)
    # Sum the last n elements of the 1D array term10
    term1 = jnp.sum(term10[n:])
    # Get the second term
    term2 = 0.5 * jnp.trace(KL)
    # Calculate T2
    assert n > 1
    T2 = (1 / (n * n * (n - 1))) * (term1 - term2)
    return T2

@jax.jit
def get_T3(K, L, n=None):
    if n is None:
        N = len(K)
        assert N % 2 == 0
        n = N // 2
    # Multiply the two kernel matrices
    KL = jnp.matmul(K, L)
    # Sum the first n columns of KL
    term10 = jnp.sum(KL[:, :n], axis=1)
    # Sum the first n elements of the 1D array term10
    term1 = jnp.sum(term10[:n])
    # Get the second term
    term2 = 0.5 * jnp.trace(KL)
    # Calculate T3
    assert n > 1
    T3 = (1 / (n * n * (n - 1))) * (term1 - term2)
    return T3

@jax.jit
def get_T4(K, L, n=None):
    if n is None:
        N = len(K)
        assert N % 2 == 0
        n = N // 2
    # Get the KL matrix
    KL = jnp.matmul(K, L)

    # Get the first term
    term1 = (jnp.sum(jnp.sum(K[:, :n], axis=1)[n:]) *
             jnp.sum(jnp.sum(L[:, :n], axis=1)[n:]))
    # Get the second term
    term2 = jnp.sum(jnp.sum(KL[:, n:], axis=1)[n:])
    # Get the third term
    term3 = jnp.sum(jnp.sum(KL[:, :n], axis=1)[:n])
    # Get the fourth term
    term4 = 0.5 * jnp.trace(KL)
    # Combine the four terms to get T4
    assert n > 1
    den = (n * (n - 1)) ** 2  # Denominator
    T4 = (1 / den) * (term1 - term2 - term3 + term4)
    return T4

@partial(jax.jit, static_argnames=('kernel_X', 'kernel_Y'))
def get_K_L_matrices(XX, YY, kernel_X=None, kernel_Y=None):
    assert kernel_X is not None and kernel_Y is not None
    N = len(XX)
    n = N // 2  # Ensure even N, which is fine for now

    # Adjust if N is odd
    N = N - N % 2  # Ensure N is even
    XX = XX[:N]
    YY = YY[:N]
    
    # Compute kernel matrices
    K = kernel_X(XX)
    L = kernel_Y(YY)

    # Set the blocks to zero using .at indexing
    K = K.at[:n, :n].set(0)
    K = K.at[n:, n:].set(0)

    L = L.at[:n, :n].set(0)
    L = L.at[n:, n:].set(0)

    return K, L


@partial(jax.jit, static_argnames=('kernel_X', 'kernel_Y'))
def get_cross_HSIC(XX, YY, kernel_X=None, kernel_Y=None, K=None, L=None):
    assert kernel_X is not None and kernel_Y is not None
    
    N = len(XX)
    
    # If N is odd, remove the last sample (batch operation)
    N = N - N % 2  # Ensure N is even
    
    # Slice arrays to even length if necessary
    XX = XX[:N]
    YY = YY[:N]
    
    # Ensure XX and YY have the same length
    assert XX.shape[0] == YY.shape[0]
    
    # Compute kernel matrices
    if K==None or L==None:
        K, L = get_K_L_matrices(XX, YY, kernel_X, kernel_Y)
    
    # Compute the four terms
    T1 = get_T1(K, L)
    T2 = get_T2(K, L)
    T3 = get_T3(K, L)
    T4 = get_T4(K, L)
    
    # Compute the Cross-HSIC statistic
    cHSIC = T1 - T2 - T3 + T4
    return cHSIC


@jax.jit
def get_w_tilde(K, L):
    N = len(K)
    n = N // 2
    
    # Ensure valid input
    if 2 * n != N:
        raise ValueError(f"Expected even number of samples, got N = {N}")
    if len(L) != N:
        raise ValueError(f"Mismatch in lengths: len(L) = {len(L)}, expected N = {N}")
    if n <= 2:
        raise ValueError("n must be greater than 2")
    
    # Compute the KL matrix
    KL = jnp.matmul(K, L)
    KL1 = jnp.sum(KL, axis=1)

    LK = jnp.matmul(L, K)
    LK1 = jnp.sum(LK, axis=1)

    K1l = jnp.sum(K[:, n:], axis=1)
    L1l = jnp.sum(L[:, n:], axis=1)
    KL1l = jnp.sum(KL[:, n:], axis=1)

    # w_tilde consists of 6 terms
    term1 = (n / (2 * (n - 1))) * jnp.sum(K * L, axis=1)
    term2 = (1 / (4 * (n - 1))) * jnp.trace(KL) * jnp.ones((2 * n,))
    term3 = (1 / (2 * (n - 1))) * (KL1 + LK1)
    term4 = 1 / (2 * (n - 1)) * K1l * L1l
    term5 = 1 / (2 * n * (n - 1)) * jnp.sum(KL1l[n:]) * jnp.ones((2 * n,))
    term6 = 1 / (4 * n * (n - 1)) * (jnp.sum(L) * K1l + jnp.sum(K) * L1l)

    # Compute w_tilde
    w_tilde = term1 + term2 - term3 - term4 - term5 + term6

    return w_tilde

@jax.jit
def get_variance(K, L, cHSIC):
    N = len(K)
    n = N // 2
    
    # Ensure valid input
    if 2 * n != N:
        raise ValueError(f"Expected even number of samples, got N = {N}")
    if len(L) != N:
        raise ValueError(f"Mismatch in lengths: len(L) = {len(L)}, expected N = {N}")
    if n <= 2:
        raise ValueError("n must be greater than 2")
    
    # Get w_tilde (using JAX-compatible function for efficient computation)
    w_tilde = get_w_tilde(K, L)
    
    # Select the first n terms of w_tilde
    w = w_tilde[:n]
    
    # Precompute constant terms
    term0 = 4 * (n - 1) / ((n - 2) ** 2)
    term1_coefficient = term0 / ((n - 1) ** 2)
    
    # Vectorized sum of w * w (element-wise multiplication)
    term1 = term1_coefficient * jnp.sum(w * w)
    
    # Compute term2 (scalar multiplication with cHSIC squared)
    term2 = term0 * n * cHSIC ** 2
    
    # Final variance calculation
    variance = term1 - term2
    
    return variance

# Now optimize the main function
@partial(jax.jit, static_argnames=('kernel_X', 'kernel_Y'))
def get_studentized_cross_hsic(XX, YY, kernel_X=None, kernel_Y=None):
    """Compute the studentized Cross-HSIC."""
    kernel_X = RBFkernel if kernel_X is None else kernel_X
    kernel_Y = RBFkernel if kernel_Y is None else kernel_Y
    assert len(XX)==len(YY)
    n = len(XX)
    
    # Compute the kernel matrices
    K, L = get_K_L_matrices(XX, YY, kernel_X, kernel_Y)
    
    # Compute the cross-HSIC statistic
    cHSIC = get_cross_HSIC(XX, YY, kernel_X, kernel_Y, K, L)
    
    # Get the variance estimate
    var = get_variance(K, L, cHSIC)
    
    # Ensure variance is non-negative using lax to handle the condition in JAX
    var = lax.cond(jnp.any(var <= 0), lambda _: 1.0, lambda _: var, operand=None)
    
    # Calculate the statistic value (studentized)
    stat = cHSIC * jnp.sqrt(n / (2 * var))  # Use jnp.sqrt for consistency
    return stat

# Fast HSIC test (already implemented, assuming crossHSIC_test exists)
def fast_hsic_test(X, Y, kernel_X=None, kernel_Y=None, alpha=0.05):
    assert isinstance(X, jnp.ndarray), "X must be a jnp array"
    assert isinstance(Y, jnp.ndarray), "Y must be a jnp array"
    kernel_X = RBFkernel if kernel_X is None else kernel_X
    kernel_Y = RBFkernel if kernel_Y is None else kernel_Y
    th = stats.norm.ppf(1-alpha) 
    stat = get_studentized_cross_hsic(X, Y, kernel_X, kernel_Y)
    return 1.0*(stat>th)