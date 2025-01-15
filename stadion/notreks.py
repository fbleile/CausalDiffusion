from functools import partial
import jax.numpy as jnp
from jax import grad, vmap
import jax.scipy.linalg

from stadion.utils import marg_indeps_to_indices

def notreks_loss(f, sigma, target_sparsity=0.1, estimator="analytic"):
    """
    Compute the notreks loss for the drift (f) and diffusion (sigma) functions of an SDE.

    Args:
        f (callable): Drift function of the SDE.
        sigma (callable): Diffusion function of the SDE.
        target_sparsity (float): Target sparsity level for the sigmoid application.
        estimator (str): Method for calculating the loss (currently only "analytic" is supported).

    Returns:
        callable: A loss function taking the inputs of `f` and `sigma` as *args.
    """

    if estimator == "analytic":
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
            # Square each entry of the Jacobian and take the mean
            W = jnp.square(jacobian_f)
            
            sparsity_threshhold = jnp.quantile(W, 1 - target_sparsity)
            # Apply the sigmoid function entrywise to introduce sparsity
            W = jax.nn.sigmoid(W - sparsity_threshhold)
            print(W)
            return W

        @partial(vmap, in_axes=(0, None, None), out_axes=0)
        def compute_loss_term(x, marg_indeps_idx, args):
            """
            Compute the notreks loss term for each input `x`.
            """
            W = compute_W(x, args)
            exp_W = jax.scipy.linalg.expm(W)
            trek_W = jnp.dot(exp_W.T, exp_W)
            
            return trek_W[marg_indeps_idx].sum()

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

    else:
        raise ValueError(f"Unknown estimator `{estimator}`.")

    return loss
