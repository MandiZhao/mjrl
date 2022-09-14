"""
This is a job script for running policy gradient algorithms on gym tasks.
Separate job scripts are provided to run few other algorithms
- For DAPG see here: https://github.com/aravindr93/hand_dapg/tree/master/dapg/examples
- For model-based NPG see here: https://github.com/aravindr93/mjrl/tree/master/mjrl/algos/model_accel
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.batch_reinforce import BatchREINFORCE
from mjrl.algos.ppo_clip import PPO
from mjrl.utils.train_agent import train_agent
import os
import json
import mj_envs
import gym
import mjrl.envs
import time as timer
import logging
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def train_loop(job_data) -> None:

    # e = GymEnv(job_data.env)
    env_name = job_data.env_name
    env_kwargs = dict(job_data.get('env_kwargs', {}))
    e = gym.make(env_name, **env_kwargs)

    # img = e.render_camera_offscreen(
    #                     sim=e.sim,
    #                     cameras=[None],
    #                     width=1024,
    #                     height=1024,
    #                     device_id=0)[0]
    logging.info('Run output saved to path: {}'.format(os.getcwd()))
    job_data.run_path = os.getcwd()
    os.makedirs('iterations', exist_ok=False)
    os.makedirs('logs', exist_ok=False)
    # breakpoint()
    job_data.wandb.name = '/'.join(
        os.getcwd().split('/')[-2:]) # date/run_name
    e = GymEnv(e)


    policy_size = tuple(eval(job_data.policy_size))
    vf_hidden_size = tuple(eval(job_data.vf_hidden_size))

    policy = MLP(e.spec, hidden_sizes=policy_size, seed=job_data.seed,
                 init_log_std=job_data.init_log_std, min_log_std=job_data.min_log_std)
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data.vf_batch_size, hidden_sizes=vf_hidden_size,
                        epochs=job_data.vf_epochs, learn_rate=job_data.vf_learn_rate)

    # Construct the algorithm
    if job_data.algorithm == 'NPG':
        # Other hyperparameters (like number of CG steps) can be specified in config for pass through
        # or default hyperparameters will be used
        agent = NPG(e, policy, baseline, normalized_step_size=job_data.rl_step_size,
                    seed=job_data.seed, save_logs=True, **job_data.alg_hyper_params,
                    env_kwargs=env_kwargs)

    elif job_data.algorithm == 'VPG':
        agent = BatchREINFORCE(e, policy, baseline, learn_rate=job_data.rl_step_size,
                            seed=job_data.seed, save_logs=True, **job_data.alg_hyper_params,
                            env_kwargs=env_kwargs)

    elif job_data.algorithm == 'NVPG':
        agent = BatchREINFORCE(e, policy, baseline, desired_kl=job_data.rl_step_size,
                            seed=job_data.seed, save_logs=True, **job_data.alg_hyper_params,
                            env_kwargs=env_kwargs)

    elif job_data.algorithm == 'PPO':
        # There are many hyperparameters for PPO. They can be specified in config for pass through
        # or defaults in the PPO algorithm will be usedc
        agent = PPO(e, policy, baseline, save_logs=True, **job_data.alg_hyper_params,
        env_kwargs=env_kwargs)
    else:
        NotImplementedError("Algorithm not found")

    logging.info("========================================")
    logging.info("Starting policy learning")
    logging.info("========================================")
    if job_data.log_wandb:
        if job_data.wandb_offline:
            logging.info('Running wandb offline')
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_API_KEY"] = job_data.wandb_api_key

        run = wandb.init(**job_data.wandb)
        run.config.update(OmegaConf.to_object(job_data))
        env_seed = env_kwargs.get('seed', 'default')
        # wandb.log(
        #     {env_name: wandb.Image(img, caption=f'envseed_{env_seed}')}
        # )

    ts = timer.time()
    train_agent(job_name='.',
                agent=agent,
                env=e,
                env_kwargs=env_kwargs,
                seed=job_data.seed,
                niter=job_data.rl_num_iter,
                gamma=job_data.rl_gamma,
                gae_lambda=job_data.rl_gae,
                num_cpu=job_data.num_cpu,
                sample_mode=job_data.sample_mode,
                num_traj=job_data.rl_num_traj,
                num_samples=job_data.rl_num_samples,
                save_freq=job_data.save_freq,
                evaluation_rollouts=job_data.eval_rollouts,
                log_wandb=job_data.log_wandb,
                stop_threshold=job_data.stop_threshold,
                )
    logging.info("========================================")
    logging.info("Job Finished. Time taken = %f" % (timer.time()-ts))
    logging.info("========================================")