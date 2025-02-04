import os
import shutil
import subprocess
from functools import partial
import time
import datetime
import traceback
from argparse import Namespace

import wandb
import jax
from jax import random, vmap
import jax.numpy as jnp
import numpy as onp

from sklearn.decomposition import PCA

from experiment.core import sample_dynamical_system, Data
from experiment.plot import plot, plot_wandb_images
from experiment.data import make_dataset, Batch, sample_batch_jax
from experiment.intervention import search_intv_theta_shift
from experiment.definitions import cpu_count, IS_CLUSTER, CONFIG_DIR

from experiment.sample import make_data

from experiment.utils.parse import load_config
from experiment.utils.version_control import get_gpu_info, get_gpu_info2
from experiment.utils.metrics import make_mse, make_wasserstein

from stadion.models import LinearSDE

def run_algo_wandb(wandb_config=None, eval_mode=False):
    """Function run by wandb.agent()"""

    # job setup
    t_init = time.time()
    exception_after_termination = None

    # wandb setup
    with wandb.init(**wandb_config):

        try:
            # this config will be set by wandb
            config = wandb.config

            # summary metrics we are interested in
            wandb.define_metric("loss", summary="min")
            wandb.define_metric("loss_fit", summary="min")
            wandb.define_metric("mse_train", summary="min")
            wandb.define_metric("mse_test", summary="min")
            wandb.define_metric("wasser_train", summary="min")
            wandb.define_metric("wasser_test", summary="min")

            wandb_run_dir = os.path.abspath(os.path.join(wandb.run.dir, os.pardir))
            print("wandb directory: ", wandb_run_dir, flush=True)

            """++++++++++++++   Data   ++++++++++++++"""
            jnp.set_printoptions(precision=2, suppress=True, linewidth=200)

            print("\nSimulating data...", flush=True)

            # load or sample data
            data_config = load_config(CONFIG_DIR / config.data_config, abspath=True)
            train_targets, test_targets, meta_data = make_data(seed=config.seed, config=data_config)

            print("done.\n", flush=True)

            """++++++++++++++   Run algorithm   ++++++++++++++"""

            _ = run_algo(train_targets, test_targets, config=config, eval_mode=eval_mode, t_init=t_init)


        except Exception as e:
            print("\n\n" + "-" * 30 + "\nwandb sweep exception traceback caught:\n")
            print(traceback.print_exc(), flush=True)
            exception_after_termination = e

        print("End of wandb.init context.\n", flush=True)

    # manual sync of offline wandb run -- this is for some reason much faster than the online mode of wandb
    print("Starting wandb sync ...", flush=True)
    subprocess.call(f"wandb sync {wandb_run_dir}", shell=True)

    # clean up
    try:
        shutil.rmtree(wandb_run_dir)
        print(f"deleted wandb directory after successful sync: `{wandb_run_dir}`", flush=True)
    except OSError as e:
        print("wandb dir not deleted.", flush=True)

    if exception_after_termination is not None:
        raise exception_after_termination

    print(f"End of run_algo_wandb after total walltime: "
          f"{str(datetime.timedelta(seconds=round(time.time() - t_init)))}",
          flush=True)

def run_algo(train_targets, test_targets, config=None, eval_mode=False, t_init=None):

    jnp.set_printoptions(precision=2, suppress=True, linewidth=200)
    key = random.PRNGKey(config.seed)
    t_run_algo = time.time()

    device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    print(f"jax backend:   {jax.default_backend()} ")
    print(f"devices:       {device_count}")
    print(f"local_devices: {local_device_count}")
    print(f"cpu_count:     {cpu_count}", flush=True)
    print(f"gpu_info:      {get_gpu_info()}", flush=True)
    print(f"               {get_gpu_info2()}", flush=True)

    if type(config) == wandb.Config:
        # wandb case
        config.update(dict(d=train_targets.data[0].shape[-1]))
    else:
        # Namespace case
        config.d = train_targets.data[0].shape[-1]

    """++++++++++++++   Low-dim visualization   ++++++++++++++"""

    if not eval_mode and config.plot_proj:
        print("Computing projection transform based on target data...")
        all_target_data = []
        for data_env in train_targets.data:
            all_target_data.append(data_env)

        # # pca
        all_target_data = onp.concatenate(all_target_data, axis=0)
        proj = PCA(n_components=2, random_state=config.seed)
        _ = proj.fit(all_target_data)
        print("done.", flush=True)

    else:
        proj = None


    """++++++++++++++   Model and parameter initialization   ++++++++++++++"""
    # theta
    key, subk = random.split(key)

    if config.model == "linear":
        model = LinearSDE(
                dependency_regularizer="NO TREKS", # Non-Structural",# "both", # 
                no_neighbors=True,
            )
    elif config.model == "mlp":
        # TODO
        pass
    else:
        raise KeyError(f"Unknown model `{config.model}`")


    """++++++++++++++   Fit Model with Data   ++++++++++++++"""
    n_train_envs = len(train_targets.data)
    
    print(f'marg_indeps: {train_targets.marg_indeps}')

    print(f"Fitting Model", flush=True)
    key, subk = random.split(key)
    model.fit(
        subk,
        train_targets.data,
        targets=train_targets.intv,
        marg_indeps=jnp.array([train_targets.marg_indeps]*n_train_envs),
        bandwidth=config.bandwidth,
        estimator="linear",
        learning_rate=config.learning_rate,
        steps=config.steps,
        batch_size=config.batch_size,
        reg=config.reg_strength,
        dep=0.1,
        warm_start_intv=True,
        verbose=10,
    )
    print(f"done.", flush=True)


    """++++++++++++++ Plot ++++++++++++++"""
    print("Starting inference...")
    
    sampler = model.sample_envs

    # MSE
    mse_accuracy = make_mse(sampler=sampler, n=config.metric_batch_size)

    # wasserstein distance
    wasser_eps_train = jnp.ones(len(train_targets.data)) * 10.
    wasser_eps_test = jnp.ones(len(test_targets.data)) * 10.

    wasserstein_accuracy_train = make_wasserstein(wasser_eps_train, sampler=sampler, n=config.metric_batch_size)
    wasserstein_accuracy_test = make_wasserstein(wasser_eps_test, sampler=sampler, n=config.metric_batch_size)

    """
    ------------------------------------
    Evaluation and logging
    ------------------------------------
    """
    
    log_dict = {}
    
    # eval metrics
    key, subk = random.split(key)
    log_dict["mse_train"], _ = \
        mse_accuracy(subk, train_targets, model.intv_param)
    
    key, subk = random.split(key)
    log_dict["wasser_train"], _ = \
        wasserstein_accuracy_train(subk, train_targets, model.intv_param)
    
    assert test_targets is not None
    
    # # assumed information about test targets
    # test_target_intv = test_targets.intv
    # test_emp_means = test_target_intv * jnp.array([data.mean(-2) for data in test_targets.data])
    
    # # init new intv_theta_test with scale 0.0
    # key, subk = random.split(key)
    # intv_theta_test_init = init_intv_theta(subk, test_target_intv.shape[0], config.d,
    #                                        scale_param=config.learn_intv_scale, scale=0.0)
    
    # # update estimate of intervention effects in test set
    # key, subk = random.split(key)
    # intv_theta_test, logs = search_intv_theta_shift(subk, theta=theta,
    #                                                 intv_theta=intv_theta_test_init,
    #                                                 target_means=test_emp_means,
    #                                                 target_intv=test_target_intv,
    #                                                 sampler=sampler,
    #                                                 n_samples=config.metric_batch_size)
    
    # to compute metrics, use test data
    key, subk = random.split(key)
    log_dict["mse_test"], _ = \
        mse_accuracy(subk, test_targets, test_targets.intv_param)
    
    key, subk = random.split(key)
    log_dict["wasser_test"], _= \
        wasserstein_accuracy_test(subk, test_targets, test_targets.intv_param)
    
    if False:
        for plot_suffix, plot_tars, plot_intv_param in [("train", train_targets, model.intv_param),
                                                        ("test", test_targets, test_targets.intv_param)]:

            # simulate rollouts
            key, subk = random.split(key)
            samples, trajs_full, _ = sampler(subk, config.metric_batch_size, intv_param=plot_intv_param, return_traj=True)

            assert samples.shape[1] >= config.plot_batch_size, "Error: should sample at least `plot_batch_size` samples "
            samples, trajs_full_single = samples[:, :config.plot_batch_size, :], trajs_full[:, 0]

            # # plot with batched target
            # key, subk = random.split(key)
            # batched_plot_tars = sample_subset(subk, plot_tars, config.plot_batch_size)

            wandb_images = plot(samples, trajs_full_single, plot_tars,
                                # title_prefix=f"{plot_suffix} t={t} ",
                                title_prefix=f"{plot_suffix}",
                                theta=model.param._store,
                                intv_theta=plot_intv_param._store,
                                true_param=train_targets.true_param,
                                ref_data=train_targets.data[0],
                                cmain="grey",
                                cfit="blue",
                                cref="grey",
                                # plot_mat=False,
                                # plot_mat=True,
                                plot_mat=plot_suffix == "train",
                                # plot_mat=plot_suffix== "train" and t == int(config.steps),
                                # plot_params=False,
                                plot_params=plot_suffix== "train" and IS_CLUSTER,
                                # plot_params=True,
                                plot_acorr=False,
                                # proj=None,
                                proj=proj,
                                # proj=proj,
                                # proj=proj if plot_suffix == "train" else None,
                                # proj=proj if t == config.steps else None,
                                # proj=proj if t == config.steps and plot_suffix == "train" else None,
                                # plot_intv_marginals=False,
                                # plot_intv_marginals=True,
                                plot_intv_marginals=IS_CLUSTER,
                                # plot_intv_marginals=not IS_CLUSTER or (plot_suffix == "train"),
                                # plot_intv_marginals=plot_suffix == "train",
                                # plot_pairwise_grid=False,
                                # plot_pairwise_grid=True,
                                # plot_pairwise_grid=not IS_CLUSTER or (plot_suffix == "train"),
                                # plot_pairwise_grid=plot_suffix== "train",
                                # plot_pairwise_grid=plot_suffix== "test",
                                plot_pairwise_grid=plot_suffix== "train",
                                # plot_pairwise_grid=plot_suffix== "train" and t == config.steps and config.d <= 5,
                                grid_type="hist-kde",
                                # contours=(0.68, 0.95),
                                # contours_alphas=(0.33, 1.0),
                                contours=(0.90,),
                                contours_alphas=(1.0,),
                                # scotts_bw_scaling=0.75,
                                scotts_bw_scaling=1.0,
                                size_per_var=1.0,
                                plot_max=config.plot_max,
                                to_wandb=IS_CLUSTER)

            if wandb_images:
                wandb_images = {f"{k}-{plot_suffix}" if "matrix" not in k else k: v
                                for k, v in wandb_images.items()}
                plot_wandb_images(wandb_images)

    print(f"End of run_algo after total walltime: "
          f"{str(datetime.timedelta(seconds=round(time.time() - t_run_algo)))}",
          flush=True)
    
    print(log_dict)
    
    return log_dict

if __name__ == "__main__":
    debug_config = Namespace()

    # fixed
    debug_config.seed = 5

    # data
    debug_config.data_config = "dev/linear.yaml"
    # debug_config.data_config = "dev/sergio.yaml"

    # model
    debug_config.model = "linear" # alternatively "mlp"

    debug_config.sampler_eps = 0.0
    debug_config.reg_eps = 0.0

    debug_config.init_scale = 0.1
    debug_config.init_diag = 0.0
    debug_config.init_intv_at_mean = True
    debug_config.learn_intv_scale = False
    debug_config.mlp_hidden = 4
    debug_config.mlp_activation = "tanh"
    debug_config.mlp_init = "uniform"
    debug_config.auto_diff = True

    # optimization
    debug_config.batch_size = 192
    debug_config.batch_size_env = 1
    debug_config.bandwidth = 5.0
    debug_config.reg_strength = 0.01
    debug_config.reg_type = "glasso"
    debug_config.grad_clip_val = None
    debug_config.force_linear_diag = True
    
    debug_config.dep_strength = 1

    debug_config.steps = 10000
    debug_config.optimizer = "adam"
    debug_config.learning_rate = 0.001

    debug_config.log_every = 100
    debug_config.eval_every = 10000
    debug_config.log_long_every = 10000
    debug_config.plot_every = 10000
    debug_config.plot_max = 3
    debug_config.metric_batch_size = 1024
    debug_config.plot_batch_size = 384
    debug_config.plot_proj = False

    debug_config.cluster_t_max = 1000

    debug_wandb_config = dict(config=debug_config, mode="disabled")
    run_algo_wandb(wandb_config=debug_wandb_config, eval_mode=False)