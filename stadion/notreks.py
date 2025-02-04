from functools import partial
import jax.numpy as jnp
from jax import grad, vmap
import jax.scipy.linalg

from stadion.utils import marg_indeps_to_indices

def no_treks(W):
    exp_W = jax.scipy.linalg.expm(W)
    trek_W = jnp.dot(exp_W.T, exp_W)
    
    return trek_W

def notreks_loss(f, sigma, target_sparsity=0.1, scale_sig=1, estimator="analytic", abs_func="abs", normalize="norm"):
    """
    Compute the notreks loss for the drift (f) and diffusion (sigma) functions of an SDE.

    Args:
        f (callable): Drift function of the SDE.
        sigma (callable): Diffusion function of the SDE.
        target_sparsity (float): Target sparsity level for the sigmoid application.
        estimator (str): Method for calculating the loss (currently only "analytic" is supported).
        abs_func (str): Method for ensuring non-negative matrix entries ("abs" or "square").
        normalize (str): Method for normalizeing matrix entries ("sigm" or "norm").

    Returns:
        callable: A loss function taking the inputs of `f` and `sigma` as *args.
    """

    if estimator == "analytic":
        # @partial(vmap, in_axes=(0, None), out_axes=0)
        def compute_W(x, args):
            """
            Compute the weighted matrix for notreks calculation.

            Args:
                x: Inputs to the drift function `f`.
                args: Parameters for `f`.
                sigma_args: Parameters for `sigma`.
            """
            # Compute the Jacobian (partial derivatives) of f with respect to x
            jacobian_f = jax.jacobian(f, argnums=0)(x, *args)
            # jacobian_f_abs = jnp.abs(jacobian_f)
            
            # jacobian_sig = jax.jacobian(sigma, argnums=0)(x, *args)
            # jacobian_sig_normed = jnp.linalg.norm(jacobian_sig, axis=1)
            
            # sig = sigma(x, *args)
            # sig_abs = jnp.abs(sig)
            
            # W = 2*jacobian_f_abs + jacobian_sig_normed + sig_abs
            
            # Square each entry of the Jacobian and take the mean
            if abs_func == "abs":
                W = jnp.abs(jacobian_f)
            elif abs_func == "square":
                W = jnp.square(jacobian_f)
            else:
                raise ValueError(f"Unknown method to ensure non-negative matrix entries `{abs_func}`.")
                
            if normalize == "sigm":
                sparsity_threshhold = jnp.quantile(W, 1 - target_sparsity)
                # Apply the sigmoid function entrywise to introduce sparsity
                W = jax.nn.sigmoid(scale_sig * (W - sparsity_threshhold))
            elif normalize == "norm":
                W = W / jnp.linalg.norm(W)
            elif normalize == "row and col norm":
                # Calculate row norms
                row_norms = jnp.linalg.norm(W, axis=1, keepdims=True)
                # Calculate column norms
                col_norms = jnp.linalg.norm(W, axis=0, keepdims=True)
                
                # Normalize each entry by its row and column norms
                return 2*W / (row_norms * col_norms)
            elif normalize == None:
                W = W
            else:
                raise ValueError(f"Unknown method to normalize matrix entries `{normalize}`.")
            
            
            return W

        @partial(vmap, in_axes=(0, None, None), out_axes=0)
        def compute_loss_term(x, marg_indeps_idx, args):
            """
            Compute the notreks loss term for each input `x`.
            """
            W = compute_W(x, args)
            no_treks_W = no_treks(W)
            #sparsity_threshhold = jnp.quantile(no_treks_W, 1 - target_sparsity)
            #no_treks_W = jax.nn.sigmoid(scale_sig * (W - sparsity_threshhold))
            return no_treks_W[marg_indeps_idx].sum()
            
            
            # W = jnp.mean(compute_W(x, args), axis=0)
            # W = W / jnp.linalg.norm(W)
            
            # no_treks_W = no_treks(W)
            
            # return no_treks_W[marg_indeps_idx].sum()

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
            
            loss_values = compute_loss_term(x, marg_indeps_idx, args)
            return loss_values.mean()
            # return loss_values

    else:
        raise ValueError(f"Unknown estimator `{estimator}`.")

    return loss


# @partial(vmap, in_axes=(0, None), out_axes=0)
# def no_treks(W, marg_indeps_idx):
#     """
#     Compute the notreks loss term for each input `x`.
#     """
#     exp_W = jax.scipy.linalg.expm(W)
#     trek_W = jnp.dot(exp_W.T, exp_W)
    
#     return trek_W[marg_indeps_idx].sum()

# @partial(vmap, in_axes=(0, None), out_axes=0)
# def compute_direct_effect_matrix(x, args):
#     """
#     Compute the weighted matrix for notreks calculation.

#     Args:
#         x: Inputs to the drift function `f`.
#         args: Parameters for `f`.
#         sigma_args: Parameters for `sigma`.
#     """
#     # Compute the Jacobian (partial derivatives) of f with respect to x
#     jacobian_f = jax.jacobian(f, argnums=0)(x, *args)
#     # Square each entry of the Jacobian and take the mean
#     W = jnp.square(jacobian_f)
    
#     sparsity_threshhold = jnp.quantile(W, 1 - target_sparsity)
#     # Apply the sigmoid function entrywise to introduce sparsity
#     W = jax.nn.sigmoid(W - sparsity_threshhold)
#     return W

# def loss(x, marg_indeps, *args):
#     """
#     Final loss function that calculates the average notreks loss.

#     Args:
#         x: Input samples to `f` and `sigma`.
#         *args: Parameters for `f` and `sigma`.

#     Returns:
#         Scalar loss value.
#     """
#     # print(f'marg_indeps in loss: {marg_indeps}')
#     marg_indeps_idx = marg_indeps_to_indices(marg_indeps)
    
#     W = compute_direct_effect_matrix(x, args)
    
#     loss_values = no_treks(W, marg_indeps_idx)
    
#     return loss_values.mean()