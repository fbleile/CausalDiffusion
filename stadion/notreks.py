from functools import partial
import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap, random
import jax.scipy.linalg
import functools

from stadion.crosshsic import get_studentized_cross_hsic, CROSS_HSIC_TH

from stadion.utils import marg_indeps_to_indices

def no_treks(W, scale = 1):
    exp_W = jax.scipy.linalg.expm(scale * W)
    trek_W = jnp.dot(exp_W.T, exp_W)
    
    return trek_W

# @functools.partial(jax.jit, static_argnums=(0,1))
def notreks_loss(model, estimator="analytic", abs_func="abs", normalize="norm"):
    """
    Compute the notreks loss for the drift (f) and diffusion (sigma) functions of an SDE.

    Args:
        f (callable): Drift function of the SDE.
        sigma (callable): Diffusion function of the SDE.
        target_sparsity (float): Target sparsity level for the sigmoid application.
        estimator (str): Method for calculating the loss (currently only "analytic" "crosshsic" is supported).
        abs_func (str): Method for ensuring non-negative matrix entries ("abs" or "square").
        normalize (str): Method for normalizeing matrix entries ("sigm" or "norm").

    Returns:
        callable: A loss function taking the inputs of `f` and `sigma` as *args.
    """
    
    f, sigma = model.f, model.sigma

    if estimator == "analytic":
        
        # @jax.jit
        @partial(vmap, in_axes=(0, None, None), out_axes=0)
        def compute_W_x(x, marg_indeps_idx, args):
            """
            Compute the weighted matrix for notreks calculation.

            Args:
                x: Inputs to the drift function `f`.
                args: Parameters for `f`.
                sigma_args: Parameters for `sigma`.
            """
            # Compute the Jacobian (partial derivatives) of f with respect to x
            jacobian_f = jax.jacobian(f, argnums=0)(x, *args)
            jacobian_f_abs = jnp.abs(jacobian_f)
            
            jacobian_sig = jax.jacobian(sigma, argnums=0)(x, *args)
            jacobian_sig_abs = jnp.linalg.norm(jacobian_sig, axis=1)
            
            sig = sigma(x, *args)
            sig_abs = jnp.abs(sig)
            
            W = jacobian_f_abs.T + jacobian_sig_abs.T + sig_abs.T
            
            h = no_treks(W / jnp.linalg.norm(W))
            
            return h[marg_indeps_idx].sum()

    else:
        raise ValueError(f"Unknown estimator `{estimator}`.")
    
    def loss(x, marg_indeps, *args):
        """
        Final loss function that calculates the average notreks loss.

        Args:
            x: Input samples to `f` and `sigma`.
            *args: Parameters for `f` and `sigma`.

        Returns:
            Scalar loss value.
        """
        # print(f'marg_indeps in loss: {marg_indeps}')
        marg_indeps_idx = marg_indeps_to_indices(marg_indeps)
        
        loss_values = compute_W_x(x, marg_indeps_idx, args)
        
        return loss_values.mean() / len(marg_indeps)

    return loss


# # @functools.partial(jax.jit, static_argnums=(0,1))
# def notreks_loss(model, estimator="analytic", abs_func="abs", normalize="norm"):
#     """
#     Compute the notreks loss for the drift (f) and diffusion (sigma) functions of an SDE.

#     Args:
#         f (callable): Drift function of the SDE.
#         sigma (callable): Diffusion function of the SDE.
#         target_sparsity (float): Target sparsity level for the sigmoid application.
#         estimator (str): Method for calculating the loss (currently only "analytic" "crosshsic" is supported).
#         abs_func (str): Method for ensuring non-negative matrix entries ("abs" or "square").
#         normalize (str): Method for normalizeing matrix entries ("sigm" or "norm").

#     Returns:
#         callable: A loss function taking the inputs of `f` and `sigma` as *args.
#     """
    
#     f, sigma = model.f, model.sigma

#     if estimator == "analytic":
        
#         # @jax.jit
#         @partial(vmap, in_axes=(0, None, None), out_axes=0)
#         def compute_W_x(x, marg_indeps, args):
#             """
#             Compute the weighted matrix for notreks calculation.

#             Args:
#                 x: Inputs to the drift function `f`.
#                 args: Parameters for `f`.
#                 sigma_args: Parameters for `sigma`.
#             """
#             # Compute the Jacobian (partial derivatives) of f with respect to x
#             jacobian_f = jax.jacobian(f, argnums=0)(x, *args)
#             jacobian_f_abs = jnp.abs(jacobian_f)
            
#             jacobian_sig = jax.jacobian(sigma, argnums=0)(x, *args)
#             jacobian_sig_normed = jnp.linalg.norm(jacobian_sig, axis=1)
            
#             sig = sigma(x, *args)
#             sig_abs = jnp.abs(sig)
            
#             W = 2*jacobian_f_abs + jacobian_sig_normed + sig_abs
            
#             return W
            
#             # # Square each entry of the Jacobian and take the mean
#             # if abs_func == "abs":
#             #     W = jnp.abs(jacobian_f)
#             # elif abs_func == "square":
#             #     W = jnp.square(jacobian_f)
#             # else:
#             #     raise ValueError(f"Unknown method to ensure non-negative matrix entries `{abs_func}`.")
                
#             # if normalize == "sigm":
#             #     sparsity_threshhold = jnp.quantile(W, 1 - target_sparsity)
#             #     # Apply the sigmoid function entrywise to introduce sparsity
#             #     W = jax.nn.sigmoid(scale_sig * (W - sparsity_threshhold))
#             # elif normalize == "norm":
#             #     W = W / jnp.linalg.norm(W)
#             # elif normalize == "row and col norm":
#             #     # Calculate row norms
#             #     row_norms = jnp.linalg.norm(W, axis=1, keepdims=True)
#             #     # Calculate column norms
#             #     col_norms = jnp.linalg.norm(W, axis=0, keepdims=True)
                
#             #     # Normalize each entry by its row and column norms
#             #     return 2*W / (row_norms * col_norms)
#             # elif normalize == None:
#             #     W = W
#             # else:
#             #     raise ValueError(f"Unknown method to normalize matrix entries `{normalize}`.")
#         def compute_W(x, marg_indeps, args):
#             return jnp.mean(compute_W_x(x, marg_indeps, args), axis=0)
            
#     elif estimator == "crosshsic":
#         # @jax.jit
#         def compute_W(x, marg_indeps, args):
#             (param, intv_param) = args
            
#             # save parameters
#             model.param = param
            
#             model.key, subk = random.split(model.key)
#             samples = model.sample(
#                 subk,
#                 x.shape[0],
#                 intv_param=intv_param,
#                 x_0=x,
#                 burnin=False,
#             )
            
#             # print(x.shape, samples.shape)
#             D = x.shape[-1]
#             W = jnp.zeros((D, D))  # Initialize D × D matrix
            
#             # Generate all index pairs (i, j) for a D x D matrix
#             indices = jnp.indices((D, D))
#             i_indices, j_indices = indices[0].flatten(), indices[1].flatten()
        
#             # Function to compute HSIC for a single (i, j) pair
#             def compute_hsic(i, j):
#                 X_i, X_j = x[:,i], samples[:,j]
#                 cross_hsic = get_studentized_cross_hsic(X_i, X_j)
#                 return cross_hsic
        
#             # Vectorize computations with jax.vmap
#             compute_hsic_vmap = jax.vmap(compute_hsic, in_axes=(0, 0))
#             hsic_values = compute_hsic_vmap(i_indices, j_indices)
            
#             W = W.at[i_indices, j_indices].set(hsic_values)
#             # print(W)
#             # sig = sigma(x, *args)
#             # sig_abs = jnp.abs(sig)
        
#             return jnn.relu(W - CROSS_HSIC_TH)

#     else:
#         raise ValueError(f"Unknown estimator `{estimator}`.")
    
#     def loss(x, marg_indeps, *args):
#         """
#         Final loss function that calculates the average notreks loss.

#         Args:
#             x: Input samples to `f` and `sigma`.
#             *args: Parameters for `f` and `sigma`.

#         Returns:
#             Scalar loss value.
#         """
#         # print(f'marg_indeps in loss: {marg_indeps}')
#         marg_indeps_idx = marg_indeps_to_indices(marg_indeps)
        
#         W = compute_W(x, marg_indeps, args)
#         W = W / jnp.linalg.norm(W)
        
#         no_treks_W = no_treks(W)
        
#         loss_values = no_treks_W[marg_indeps_idx].sum()
        
#         return loss_values

#     return loss