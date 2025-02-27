import time
import datetime
import traceback
import copy
import pandas as pd
import wandb
import os

import torch

from multiprocessing import Pool, get_context

from tabulate import tabulate

import jax
from jax import random
import jax.numpy as jnp

from intervention import search_intv_theta_shift
from definitions import cpu_count, CONFIG_DIR

from sample import make_data

from utils.parse import load_config, expand_config
from utils.version_control import get_gpu_info, get_gpu_info2
from utils.metrics import make_mse, make_wasserstein
from utils.NN_dim_bandwidth import NeuralNetwork

from stadion.models import LinearSDE
from stadion.parameters import InterventionParameters

def wandb_run_algo(key, train_targets, test_targets, config=None, t_init=None, log_dict = {}):

    jnp.set_printoptions(precision=2, suppress=True, linewidth=200)
    t_run_algo = time.time()

    """++++++++++++++   Model and parameter initialization   ++++++++++++++"""
    # theta
    key, subk = random.split(key)

    if config["model"] == "linear":
        model = LinearSDE(
                subk,
                dependency_regularizer=config["inductive_bias"]["dependency_regularizer"], # "SampleBackprop", # "SampleCrossHSIC", # "Lyapunov",# "both", # 
                no_neighbors=config["inductive_bias"]["no_neighbors"],
                sde_kwargs = {key: value for key, value in config["sde"].items()} if "sde" in config else None,
            )
    elif config["model"] == "mlp":
        # TODO
        pass
    else:
        raise KeyError(f"Unknown model `{config['model']}`")

    """++++++++++++++   Fit Model with Data   ++++++++++++++"""
    n_train_envs = len(train_targets.data)

    print(f"Fitting Model", flush=True)
    print(f"{config}")
    key, subk = random.split(key)
    model.fit(
        subk,
        train_targets.data,
        targets=train_targets.intv,
        marg_indeps=jnp.array([train_targets.marg_indeps]*n_train_envs),
        bandwidth=config["bandwidth"],
        objective_fun=config["objective_fun"],
        estimator=config["estimator"],
        learning_rate=config["learning_rate"],
        steps=config["steps"],
        batch_size=config["batch_size"],
        reg=config["reg_strength"],
        dep=config["inductive_bias"]["dep_strength"],
        warm_start_intv=True,
        verbose=10,
        k_reg=config["kernel_reg"]
    )
    print(f"done.", flush=True)
    
    """
    ------------------------------------
    Evaluation and logging
    ------------------------------------
    """
    
    log_dict['train total time'] = time.time() - t_run_algo
    
    print("Starting inference...")
    
    sampler = model.sample_envs

    # MSE
    mse_accuracy = make_mse(sampler=sampler, n=config["metric_batch_size"])

    # wasserstein distance
    wasser_eps_train = jnp.ones(len(train_targets.data)) * 10.
    wasser_eps_test = jnp.ones(len(test_targets.data)) * 10.

    wasserstein_accuracy_train = make_wasserstein(wasser_eps_train, sampler=sampler, n=config["metric_batch_size"])
    wasserstein_accuracy_test = make_wasserstein(wasser_eps_test, sampler=sampler, n=config["metric_batch_size"])
    
    assert test_targets is not None
    
    # assumed information about test targets
    test_target_intv = test_targets.intv
    test_emp_means = test_target_intv * jnp.array([data.mean(-2) for data in test_targets.data])
    
    # update estimate of intervention effects in test set
    key, subk = random.split(key)
    intv_theta_test, logs = search_intv_theta_shift(subk,
                                                    theta=model.param._store,
                                                    intv_param=test_targets.intv_param,
                                                    target_means=test_emp_means,
                                                    target_intv=test_target_intv,
                                                    sampler=sampler,
                                                    n_samples=config["metric_batch_size"])
    intv_theta_test = InterventionParameters(
            parameters= intv_theta_test,
            targets=test_targets.intv_param.targets
        )
        
    # eval metrics
    key, subk = random.split(key)
    log_dict["avg_mse_train"], log_dict["med_mse_train"], log_dict["full_mse_train"] = \
        mse_accuracy(subk, train_targets, model.intv_param)
    
    key, subk = random.split(key)
    log_dict["avg_wasser_train"], log_dict["med_wasser_train"], log_dict["full_wasser_train"] = \
        wasserstein_accuracy_train(subk, train_targets, model.intv_param)
    
    # to compute metrics, use test data
    key, subk = random.split(key)
    log_dict["avg_mse_test"], log_dict["med_mse_test"], log_dict["full_mse_test"] = mse_accuracy(subk, test_targets, intv_theta_test)
    
    # key, subk = random.split(key)
    # log_dict["wasser_test_my"], _= wasserstein_accuracy_test(subk, test_targets, test_targets.intv_param)
    key, subk = random.split(key)
    log_dict["avg_wasser_test"], log_dict["med_wasser_test"], log_dict["full_wasser_test"]= wasserstein_accuracy_test(subk, test_targets, intv_theta_test)
    
    print(f"End of run_algo after total walltime: "
          f"{str(datetime.timedelta(seconds=round(time.time() - t_run_algo)))}",
          flush=True)

    table = [
        ["Avg MSE", f"{log_dict['avg_mse_train']:.3f}", f"{log_dict['avg_mse_test']:.3f}"],
        ["Med MSE", f"{log_dict['med_mse_train']:.3f}", f"{log_dict['med_mse_test']:.3f}"],
        ["Avg Wasserstein", f"{log_dict['avg_wasser_train']:.3f}", f"{log_dict['avg_wasser_test']:.3f}"],
        ["Med Wasserstein", f"{log_dict['med_wasser_train']:.3f}", f"{log_dict['med_wasser_test']:.3f}"]
    ]

    print(tabulate(table, headers=["Metric", "Train Set", "Test Set"], tablefmt="grid"))
    print(f"\nTotal Train Time: {log_dict['train total time']:.3f} s")
    
    return log_dict

def single_debug_run(test = False):
    print("single_debug_run")
    debug_config = {}

    # fixed
    debug_config["seed"] = 60

    # data
    # debug_config.data_config = "/Users/bleile/Master/Thesis Work/CausalDiffusion/config/dev/linear10.yaml"
    debug_config["data_config"] = "/Users/bleile/Master/Thesis Work/CausalDiffusion/config/dev/linear20.yaml"

    
    print(debug_config)
    # debug_config.data_config = "dev/sergio.yaml"

    # model
    debug_config["model"] = "linear" # alternatively "mlp"
    
    debug_config["objective_fun"] = "skds"
    debug_config["estimator"] = "linear"
    
    debug_config["inductive_bias"] = {
            "dependency_regularizer": "None",
            "dep_strength": 10,
            "estimator": "analytic",
            "no_neighbors": False
        }

    # optimization
    debug_config["batch_size"] = 192
    debug_config["bandwidth"] = 1.0 # 5.0
    debug_config["reg_strength"] = 0.1
    
    debug_config["dep_strength"] = 10

    debug_config["steps"] = 7000
    debug_config["learning_rate"] = 0.001

    debug_config["metric_batch_size"] = 1024

    config = debug_config
    
    key = random.PRNGKey(config["seed"])
    
    t_init = time.time()

    try:
        """++++++++++++++   Hardware   ++++++++++++++"""
        device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        print(f"jax backend:   {jax.default_backend()} ")
        print(f"devices:       {device_count}")
        print(f"local_devices: {local_device_count}")
        print(f"cpu_count:     {cpu_count}", flush=True)
        print(f"gpu_info:      {get_gpu_info()}", flush=True)
        print(f"               {get_gpu_info2()}", flush=True)
        
        """++++++++++++++   Data   ++++++++++++++"""
        jnp.set_printoptions(precision=2, suppress=True, linewidth=200)

        print("\nSimulating data...", flush=True)

        # load or sample data
        key, subk = random.split(key)
        data_config = load_config(CONFIG_DIR / config["data_config"], abspath=True)
        train_targets, test_targets, meta_data = make_data(key=subk, config=data_config)

        print("done.\n", flush=True)

        """++++++++++++++   Run algorithm   ++++++++++++++"""
        key, subk = random.split(key)
        _ = wandb_run_algo(subk, train_targets, test_targets, config=config, t_init=t_init, log_dict=config)

    except Exception:
        print(traceback.print_exc(), flush=True)
        
def hyperparam_tuning(seed, data_config_str = None, model_config_str = None):
    
    model_master_config = load_config(model_config_str, abspath=True)
    model_master_expanded_configs = sum((expand_config(model_master_config[key]) for key in model_master_config), [])
    
    # Print expanded configurations
    for i, conf in enumerate(model_master_expanded_configs, 1):
        print(f"Config {i}:", conf)

    key = random.PRNGKey(seed)
    
    t_init = time.time()

    try:
        """++++++++++++++   Hardware   ++++++++++++++"""
        device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        print(f"jax backend:   {jax.default_backend()} ")
        print(f"devices:       {device_count}")
        print(f"local_devices: {local_device_count}")
        print(f"cpu_count:     {cpu_count}", flush=True)
        print(f"gpu_info:      {get_gpu_info()}", flush=True)
        print(f"               {get_gpu_info2()}", flush=True)
        
        """++++++++++++++   Data   ++++++++++++++"""
        jnp.set_printoptions(precision=2, suppress=True, linewidth=200)

        print("\nSimulating data...", flush=True)

        # load or sample data
        data_config = load_config(data_config_str, abspath=True)
        
        datasets = {}
        
        logger = {}
        logger["id"] = data_config["id"]
        logger["n_vars"] = data_config["n_vars"]
        logger["sparsity"] = data_config["sparsity"]
        logger["marg_indeps"] = data_config["marg_indeps"]
        
        for i in range(data_config["n_datasets"]):
            key, subk = random.split(key)
            train_targets, test_targets, meta_data = make_data(key=subk, config=data_config)
            datasets[i] = {}
            datasets[i]["train_targets"] = train_targets
            datasets[i]["test_targets"] = test_targets
            datasets[i]["meta_data"] = meta_data

        print("done.\n", flush=True)

        """++++++++++++++   Run algorithm   ++++++++++++++"""
        logs = []
        for i, config in enumerate(model_master_expanded_configs):
            model_log = {}
            model_log["model"] = config["model"]
            model_log["objective_fun"] = config["objective_fun"]
            model_log["bandwidth"] = config["bandwidth"]
            model_log["steps"] = config["steps"]
            model_log["model"] = config["model"]
            model_log["inductive_bias"] = config["inductive_bias"]
            for j, value in datasets.items():
                train_targets, test_targets, _ = datasets[j].values()
                model_log_copy = copy.deepcopy(model_log)
                
                key, subk = random.split(key)
                data_log = wandb_run_algo(subk, train_targets, test_targets, config=config, t_init=t_init)
                model_log_copy.update(data_log)
                
                logs.append(model_log_copy)
       
        # Convert list of dictionaries to a DataFrame
        df = pd.DataFrame(logs)
        
        # Write to a CSV file
        df.to_csv('output.csv', index=False)  # index=False avoids writing row indices
    
    except Exception:
        print(traceback.print_exc(), flush=True)


# This is now a top-level function that can be serialized properly
def run_single_config(config_and_key):
    # Unpack the config and PRNG key
    config, key, dataset, model_dim_bandwidth = config_and_key
    
    train_targets, test_targets, _ = dataset.values()
    config["d"] = train_targets.data[0].shape[-1]

    # Prepare input for the model
    input_data = torch.tensor([[config["d"], config["bandwidth"]]], dtype=torch.float32)
    
    # Make the prediction
    with torch.no_grad():
        avg_kernel = model_dim_bandwidth(input_data).item()
        config["kernel_reg"] = (avg_kernel)**(-1)

    
    model_log = {
        "model": config["model"],
        "objective_fun": config["objective_fun"],
        "bandwidth": config["bandwidth"],
        "steps": config["steps"],
        "inductive_bias": config["inductive_bias"]
    }

    model_log_copy = copy.deepcopy(model_log)

    # Split key for each configuration
    key, subk = random.split(key)  # Generate new key for each run
    data_log = wandb_run_algo(subk, train_targets, test_targets, config=config, t_init=time.time())
    model_log_copy.update(data_log)

    return model_log_copy

# This is the main function that initializes everything and runs the parallelized hyperparameter tuning
def hyperparam_tuning_wandb(seed, data_config_str=None, model_config_str=None):
    model_master_config = load_config(model_config_str, abspath=True)
    model_master_expanded_configs = sum((expand_config(model_master_config[key]) for key in model_master_config), [])

    # Initialize PRNG key
    key = random.PRNGKey(seed)

    try:
        # Initialize Weights & Biases
        wandb.init(project="your_project_name", config={"seed": seed, "data_config": data_config_str, "model_config": model_config_str})
        # wandb.init(sync_tensorboard=False, reinit=True)

        """++++++++++++++   Hardware ++++++++++++++"""
        device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        print(f"jax backend:   {jax.default_backend()}")
        print(f"devices:       {device_count}")
        print(f"local_devices: {local_device_count}")
        
        """++++++++++++++ Load kernel regularizer Neural Network ++++++++++++++"""
        # Define relative path from the current working directory
        # relative_path = os.path.join(os.path.dirname(__file__), 'utils', 'dim_bandwidth_model.pth')
        
        # Load the model
        model_dim_bandwidth = torch.load('/Users/bleile/Master/Thesis Work/CausalDiffusion/utils/dim_bandwidth_model.pth', weights_only=False)
        model_dim_bandwidth.eval()
        
        """++++++++++++++   Data ++++++++++++++"""
        print("\nSimulating data...", flush=True)
        data_config = load_config(data_config_str, abspath=True)

        datasets = {}
        data_model_log = {
            "id": data_config["id"],
            "n_vars": data_config["n_vars"],
            "sparsity": data_config["sparsity"],
            "marg_indeps": data_config["marg_indeps"]
        }

        for i in range(data_config["n_datasets"]):
            key, subk = random.split(key)
            train_targets, test_targets, meta_data = make_data(key=subk, config=data_config)
            datasets[i] = {"train_targets": train_targets, "test_targets": test_targets, "meta_data": meta_data}

        print("done.\n", flush=True)

        """++++++++++++++   Run algorithm ++++++++++++++"""
        logs = []
        
        # Prepare the data to pass to the Pool
        config_and_key_and_dataset_list = [
            (config, key, datasets[i], model_dim_bandwidth) for i in datasets for config in model_master_expanded_configs
        ]
        
        # # Parallelize the runs with controlled parallelism
        # with get_context("spawn").Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as p:
        #     logs = p.map(run_single_config, config_and_key_and_dataset_list)
        
        with get_context("spawn").Pool(processes=1, maxtasksperchild=1) as p:
            for i, config in enumerate(config_and_key_and_dataset_list, 1):
                log = p.apply(run_single_config, args=(config,))
                logs.append(log)
                print(f'Finished {i} / {len(config_and_key_and_dataset_list)}')

        # Add seed and W&B execution name to the beginning of each log
        logs_with_config = [
            {"seed": seed, "wandb_name": wandb.run.name, **log, **data_model_log} for log in logs
        ]
    except Exception:
        print(traceback.print_exc(), flush=True)
    finally:
        wandb.finish()
    return logs_with_config

if __name__ == "__main__":
    
    data_config_str = "/Users/bleile/Master/Thesis Work/CausalDiffusion/config/dev/linear20_SCM.yaml"
    model_master_config_str = "/Users/bleile/Master/Thesis Work/CausalDiffusion/config/dev/models.yaml"
    
    logs = hyperparam_tuning_wandb(113, data_config_str, model_master_config_str)
    
    df = pd.DataFrame(logs)
    df.replace({r'\n': ' ', r'\r': ' '}, regex=True, inplace=True)
    df.to_csv('output.csv', mode='a', header=not pd.io.common.file_exists('output.csv'), index=False)