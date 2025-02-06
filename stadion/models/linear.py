import functools

from jax import random, vmap
import jax.numpy as jnp
import jax.lax as lax
from scipy.linalg import solve_continuous_lyapunov

from stadion.parameters import ModelParameters, InterventionParameters
from stadion.sde import SDE
from stadion.inference import KDSMixin
from stadion.utils import to_diag, tree_global_norm, tree_init_normal, \
    marg_indeps_to_indices, is_hurwitz_stable, get_one_hot_index
from stadion.notreks import notreks_loss


class LinearSDE(KDSMixin, SDE):
    """
    Linear SDE with shift-scale intervention model.

    The drift function is a linear function and the diffusion function
    is a constant diagonal matrix. The interventions shift the drift and
    scale the diffusion matrix.

    Args:
        sparsity_regularizer (str, optional): Type of sparsity regularizer to use.
            Implemented are: ``outgoing,``ingoing``,``both`.`
        dependency_regularizer: (str, optional) decides on the method to penalize dependence.
            ``NO TREKS,``Lyapunov``,``both``, ``None``.
        no_neighbors: (bool, optional) masks dependency of functions for
            ``independent`` variables.
        sde_kwargs (dict, optional): any keyword arguments passed to ``SDE`` superclass.

    """

    def __init__(
        self,
        key,
        sparsity_regularizer="both",
        dependency_regularizer="None",
        no_neighbors=False,
        sde_kwargs=None,
    ):

        sde_kwargs = sde_kwargs or {}
        SDE.__init__(self, **sde_kwargs)

        self.sparsity_regularizer = sparsity_regularizer
        self.dependency_regularizer = dependency_regularizer
        self.key, subk = random.split(key)
        self.notreks_loss = notreks_loss(self) if dependency_regularizer in ("both", "NO TREKS") else None
        self.no_neighbors = no_neighbors
        self.marg_indeps = None


    def init_param(self, d, scale=1e-6, fix_speed_scaling=True, marg_indeps=None):
        """
        Samples random initialization of the SDE model parameters.
        See :func:`~stadion.inference.KDSMixin.init_param`.
        """
        
        shape = {
            "weights": jnp.zeros((d, d)),
            "biases": jnp.zeros((d,)),
            "log_noise_scale": -2 * jnp.ones((d,)),
        }
        
        self.key, subk = random.split(self.key)
        param = tree_init_normal(subk, shape, scale=scale)
        
        self.marg_indeps = marg_indeps
        
        diag_idx = jnp.diag_indices(d)
        marg_row_idx, marg_col_idx = marg_indeps_to_indices(self.marg_indeps)
        marg_indeps_idx = jnp.stack(\
                                    [jnp.concatenate([marg_row_idx, marg_col_idx]),\
                                     jnp.concatenate([marg_col_idx, marg_row_idx])], axis=1)
        # Convert diag_idx (tuple of arrays) to a suitable format for concatenation
        diag_idx_merged = jnp.stack(diag_idx, axis=1)  # Convert to 2D array with shape (d, 2)
        
        # Merge the index sets
        idx_merged = jnp.concatenate([diag_idx_merged, marg_indeps_idx], axis=0)
        
        # Merge the values
        merged_values = jnp.concatenate([jnp.full((diag_idx_merged.shape[0],), -1.0), jnp.full((marg_indeps_idx.shape[0],), 0.0)])
        
        if fix_speed_scaling and self.no_neighbors:
            return ModelParameters(
                parameters=param,
                fixed={"weights": tuple(array.astype(int) for array in idx_merged.T)},
                fixed_values={"weights": merged_values},
            )
        elif fix_speed_scaling:
            return ModelParameters(
                parameters=param,
                fixed={"weights": diag_idx},
                fixed_values={"weights": -1.0},
            )
        elif self.no_neighbors:
            return ModelParameters(
                parameters=param,
                fixed={"weights": (marg_row_idx, marg_col_idx)},
                fixed_values={"weights": 0.0},
            )
        else:
            return ModelParameters(
                parameters=param,
            )


    def init_intv_param(self, d, n_envs=None, scale=1e-6, targets=None, x=None):
        """
        Samples random initialization of the intervention parameters.
        See :func:`~stadion.inference.KDSMixin.init_intv_param`.
        """
        # pytree of [n_envs, d, ...]
        # intervention effect parameters
        vec_shape = (n_envs, d) if n_envs is not None else (d,)
        shape = {
            "shift": jnp.zeros(vec_shape),
            "log_scale": jnp.zeros(vec_shape),
        }
        self.key, subk = random.split(self.key)
        intv_param = tree_init_normal(subk, shape, scale=scale)

        # if provided, store intervened variables for masking
        if targets is not None:
            targets = jnp.array(targets, dtype=jnp.float32)
            assert targets.shape == vec_shape

        # if provided, warm-start intervention shifts of the target variables
        if x is not None and targets is not None and n_envs is not None:
            assert len(x) == n_envs
            assert all([data.shape[-1] == d] for data in x)
            ref = x[0].mean(-2)
            mean_shift = jnp.array([jnp.where(targets_, (x_.mean(-2) - ref), jnp.array(0.0)) for x_, targets_ in zip(x, targets)])
            intv_param["shift"] += mean_shift

        return InterventionParameters(parameters=intv_param, targets=targets)

    """
    Model
    """

    def f_j(self, x, param):
        w = param["weights"]
        b = param["biases"]
        assert w.shape[0] == x.shape[-1] and w.ndim == 1 and b.ndim == 0
        return x @ w + b


    def f(self, x, param, intv_param):
        """
        Linear drift :math:`f(\\cdot)` with shift-scale intervention model.
        See :func:`~stadion.sde.SDE.f`.
        """
        # compute drift scalar f(x)_j for each dx_j using the input x
        f_vec = vmap(self.f_j, in_axes=(None, 0), out_axes=-1)(x, param)
        assert x.shape == f_vec.shape

        # intervention: shift f(x) by scalar
        # [d,]
        if intv_param is not None:
            # print(f'intv_param {intv_param["shift"].shape}')
            # print(f'f_vec {f_vec.shape}')
            f_vec += intv_param["shift"]
        assert x.shape == f_vec.shape
        return f_vec


    def sigma(self, x, param, intv_param):
        """
        Diagonal constant :math:`\\sigma(\\cdot)` with shift-scale intervention model
        See :func:`~stadion.sde.SDE.sigma`.
        """
        d = x.shape[-1]

        # compute sigma(x)
        c = jnp.exp(param["log_noise_scale"])
        sig_mat = to_diag(jnp.ones_like(x)) * c
        assert sig_mat.shape == (*x.shape, x.shape[-1])

        # intervention: scale sigma by scalar
        # [d,]
        if intv_param is not None:
            scale = jnp.exp(intv_param["log_scale"])
            sig_mat = jnp.einsum("...ab,a->...ab", sig_mat, scale)

        assert sig_mat.shape == (*x.shape, d)
        return sig_mat


    """
    Inference functions
    """

    @staticmethod
    def _regularize_ingoing(param):
        """
        Group LASSO regularization term that L1-penalizes (sparsifies) ingoing causal dependencies
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        d = param["weights"].shape[0]

        def group_lasso_j(param_j, j):
            # [d,] compute L2 norm for each group (each causal parent)
            group_lasso_terms_j = vmap(functools.partial(tree_global_norm, p=2.0), 0, 0)(param_j["weights"])

            # mask self-influence
            group_lasso_terms_j = jnp.where(jnp.eye(d)[j], 1e-20, group_lasso_terms_j)

            # [] compute Lp group lasso (in classical group lasso, p=1)
            lasso = tree_global_norm(group_lasso_terms_j, p=1.0)
            return lasso

        # group lasso for each causal mechanism
        reg_w1 = vmap(group_lasso_j, (0, 0), 0)(param, jnp.arange(d)).mean(0)
        return reg_w1


    @staticmethod
    def _regularize_outgoing(param):
        """
        Group LASSO regularization term that L1-penalizes (sparsifies) outgoing causal dependencies
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        d = param["weights"].shape[0]

        # group lasso that groups not by causal mechanism but by outgoing dependencies (masking self-influence)
        # [d,] compute L2 norm for each group (axis = 1 since we want to group by outgoing dependencies)
        groupf = vmap(functools.partial(tree_global_norm, p=2.0), 1, 0)
        group_lasso_terms_j = groupf(param["weights"] * (1 - jnp.eye(d)))

        # [] compute Lp group lasso (in classical group lasso, p=1) and scale by number of variables
        lasso = tree_global_norm(group_lasso_terms_j, p=1.0) / d
        return lasso

    def regularize_sparsity(self, param):
        """
        Sparsity regularization.
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """
        if self.sparsity_regularizer == "ingoing":
            reg = LinearSDE._regularize_ingoing(param)
        elif self.sparsity_regularizer == "outgoing":
            reg = LinearSDE._regularize_outgoing(param)
        elif self.sparsity_regularizer == "both":
            reg = LinearSDE._regularize_ingoing(param)\
                + LinearSDE._regularize_outgoing(param)
        else:
            raise ValueError(f"Unknown regularizer `{self.regularize_sparsity}`")
        return reg
    
    @staticmethod
    def compute_stationary_covar_mat(M, D):
        """
        Compute Σ (stationary covariance matrix) using the formula:
        vec(Σ) = -(I_n ⊗ M + M ⊗ I_n)^(-1) vec(DD^T)
        
        Args:
            M (jax.numpy.ndarray): Matrix M of shape (n, n).
            D (jax.numpy.ndarray): Matrix D of shape (n, n).
        
        Returns:
            jax.numpy.ndarray: Stationary covariance matrix Σ of shape (n, n).
        """
        n = M.shape[0]
        I_n = jnp.eye(n)  # Identity matrix of size n
    
        # Compute Kronecker products
        kron_1 = jnp.kron(I_n, M)
        kron_2 = jnp.kron(M, I_n)
    
        # Construct the matrix to invert
        A = kron_1 + kron_2
    
        # Compute vec(DD^T)
        DD_T = D @ D.T
        vec_DD_T = jnp.reshape(DD_T, (-1,))
    
        # Solve for vec(Σ)
        vec_sigma = -jnp.linalg.solve(A, vec_DD_T)
    
        # Reshape vec(Σ) back into Σ
        Sigma = jnp.reshape(vec_sigma, (n, n))
    
        return Sigma
    
    @staticmethod
    def regularize_dependence_lyapunov(marg_indeps, param, intv_param):
        M = param["weights"]
        is_stable = is_hurwitz_stable(M)
        
        d = int(M.shape[0])
        
        
        # compute sigma(x)
        c = jnp.exp(param["log_noise_scale"])
        sig_mat = to_diag(jnp.ones((d,))) * c
        assert sig_mat.shape == (d, d)

        # intervention: scale sigma by scalar
        # [d,]
        if intv_param is not None:
            scale = jnp.exp(intv_param["log_scale"])
            sig_mat = jnp.einsum("...ab,a->...ab", sig_mat, scale)
        
        # Compute the stationary covariance matrix Γ by solving the Lyapunov equation
        Gamma = LinearSDE.compute_stationary_covar_mat(M, -sig_mat @ sig_mat.T)
        
        pen = 0.0
        epsilon = 1e-8  # Small value to avoid division by zero
    
        for i, j in marg_indeps:
            def compute_term():
                return (Gamma[i, j] / (Gamma[i, i] * Gamma[j, j] + epsilon))**2
    
            # Only add the penalty if Gamma[i, j] is not zero
            term = lax.cond(Gamma[i, j] != 0, compute_term, lambda: 0.0)
            pen += term
    
        return pen * is_stable
    
    @staticmethod
    def regularize_dependence_no_treks(loss, x, marg_indeps, param, intv_param):
        assert notreks_loss != None
        if marg_indeps is None or len(marg_indeps) == 0:
            return 0
        # jnp.array([[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.]])
        dep_loss = loss(x, marg_indeps, param, intv_param)
        return dep_loss
    
    def regularize_dependence(self, x, marg_indeps, param, intv_param):
        # ``NO TREKS,``Non-Structural``,``both``, ``None``.
        if self.dependency_regularizer == "None":
            return 0
        elif self.dependency_regularizer == "NO TREKS":
            reg = LinearSDE.regularize_dependence_no_treks(self.notreks_loss, x, marg_indeps, param, intv_param)
        elif self.dependency_regularizer == "Lyapunov":
            reg = LinearSDE.regularize_dependence_lyapunov(marg_indeps, param, intv_param)
        elif self.dependency_regularizer == "both":
            reg = LinearSDE.regularize_dependence_lyapunov(marg_indeps, param, intv_param)\
                + LinearSDE.regularize_dependence_no_treks(self.notreks_loss, x, marg_indeps, param, intv_param)
        else:
            raise ValueError(f"Unknown dependence regularizer `{self.dependency_regularizer}`")
        
        # self.param = param
        # self.key, subk = random.split(self.key)
        # self.sample(subk, x.shape[0], intv_param=intv_param)
        
        return reg



