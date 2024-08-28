# system operations
import inspect
import os
import uuid
import warnings

# date and time
import datetime

# type hinting
from typing import Any

# User interaction
from ipywidgets import Button, HTML
from ipywidgets import Text, HBox, VBox

# data visualization
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm

# data manipulation
from bs4 import BeautifulSoup
import math
import numpy as np
import pandas as pd
import random
import re
import requests
import simplejson as json

# utilis functions
# from utilis import select_buildings, select_simulation_period, get_kpis, plot_battery_soc_profiles, plot_building_load_profiles, 
from utilis import *

# cityLearn
from citylearn.agents.base import (
    BaselineAgent,
    Agent as RandomAgent
)
from citylearn.agents.rbc import HourRBC
from citylearn.agents.q_learning import TabularQLearning
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import (
    NormalizedObservationWrapper,
    StableBaselines3Wrapper,
    TabularQLearningWrapper
)

# RL algorithms
from stable_baselines3 import SAC
print("*********************Environment Requirements Check Finished: Success*****************")


#  Patameters setting, random seeds
RANDOM_SEED = 0
print('Random seed:', RANDOM_SEED)
BUILDING_COUNT = 2
DAY_COUNT = 7
building_name = 'Building_1'
DATASET_NAME = 'citylearn_challenge_2023_phase_1'
CENTRAL_AGENT = True
ACTIVE_OBSERVATIONS = [
    'hour',
    'day_type',
    'solar_generation',
    'net_electricity_consumption',
    'electrical_storage_soc',
    'month',
    'outdoor_dry_bulb_temperature',
    'outdoor_relative_humidity',
    'diffuse_solar_irradiance',
    'non_shiftable_load',
    'electricity_pricing'
]
# print('All CityLearn datasets:', sorted(DataSet.get_names()))
schema = DataSet.get_schema(DATASET_NAME)
root_directory = schema['root_directory']
result_save_dir = '/data/mengxin/ping/CityLearn/results/2023/finetuned_SAC_more_observation_lr6e4'
# Check if the directory exists
if not os.path.exists(result_save_dir):
    # If it doesn't exist, create the directory
    os.makedirs(result_save_dir)
    print(f"Directory {result_save_dir} created.")
else:
    print(f"Directory {result_save_dir} already exists.")



### Print selected building sections
BUILDINGS = select_buildings(
    DATASET_NAME,
    BUILDING_COUNT,
    RANDOM_SEED,
)
print('Selected building:', BUILDINGS)

SIMULATION_START_TIME_STEP, SIMULATION_END_TIME_STEP = select_simulation_period(
    DATASET_NAME,
    DAY_COUNT,
    RANDOM_SEED
)

print(
    f'Selected {DAY_COUNT}-day simulation period:',
    (SIMULATION_START_TIME_STEP, SIMULATION_END_TIME_STEP)
)


env = CityLearnEnv(
    DATASET_NAME,
    central_agent=CENTRAL_AGENT,
    buildings=BUILDINGS,
    active_observations=ACTIVE_OBSERVATIONS,
    simulation_start_time_step=SIMULATION_START_TIME_STEP,
    simulation_end_time_step=SIMULATION_END_TIME_STEP,
)

print('Current time step:', env.time_step)
print('environment number of time steps:', env.time_steps)
print('environment uses central agent:', env.central_agent)
print('Number of buildings:', len(env.buildings))

# electrical storage
print('Electrical storage capacity:', {
    b.name: b.electrical_storage.capacity for b in env.buildings
})
print('Electrical storage nominal power:', {
    b.name: b.electrical_storage.nominal_power for b in env.buildings
})
print('Electrical storage loss_coefficient:', {
    b.name: b.electrical_storage.loss_coefficient for b in env.buildings
})
print('Electrical storage soc:', {
    b.name: b.electrical_storage.soc[b.time_step] for b in env.buildings
})
print('Electrical storage efficiency:', {
    b.name: b.electrical_storage.efficiency for b in env.buildings
})
print('Electrical storage electricity consumption:', {
    b.name: b.electrical_storage.electricity_consumption[b.time_step]
    for b in env.buildings
})
print('Electrical storage capacity loss coefficient:', {
    b.name: b.electrical_storage.capacity_loss_coefficient for b in env.buildings
})
print()
# pv
print('PV nominal power:', {
    b.name: b.pv.nominal_power for b in env.buildings
})
print()
# active observations
print('Active observations:', {b.name: b.active_observations for b in env.buildings})
# active actions
print('Active actions:', {b.name: b.active_actions for b in env.buildings})

#######BASELINE############ 
baseline_env = CityLearnEnv(
                DATASET_NAME,
                central_agent=CENTRAL_AGENT,
                buildings=BUILDINGS,
                active_observations=ACTIVE_OBSERVATIONS,
                simulation_start_time_step=SIMULATION_START_TIME_STEP,
                simulation_end_time_step=SIMULATION_END_TIME_STEP,
                )
baseline_model = BaselineAgent(baseline_env)

# always start by reseting the environment
observations, _ = baseline_env.reset()

# step through the environment until terminal
# state is reached i.e., the control episode ends
while not baseline_env.terminated:
    # select actions from the model
    actions = baseline_model.predict(observations)

    # apply selected actions to the environment
    observations, _, _, _, _ = baseline_env.step(actions)

########save the v.unwrapped.net_electricity_consumption data:
net_electricity_consumption = baseline_env.unwrapped.buildings[0].net_electricity_consumption
# 定义保存文件的路径
save_file_path = os.path.join(result_save_dir, f'net_electricity_consumption_{SIMULATION_START_TIME_STEP}_{SIMULATION_END_TIME_STEP}.npy')

# 将数据保存为 numpy 文件
np.save(save_file_path, net_electricity_consumption)

print(f'Building {BUILDINGS[0]} (editable) Net electricity consumption data saved to {save_file_path}')

    
#######SAC###################
sac_env = CityLearnEnv(
    DATASET_NAME,
    central_agent=CENTRAL_AGENT,
    buildings=BUILDINGS,
    active_observations=ACTIVE_OBSERVATIONS,
    simulation_start_time_step=SIMULATION_START_TIME_STEP,
    simulation_end_time_step=SIMULATION_END_TIME_STEP,
)
sac_env = NormalizedObservationWrapper(sac_env)
sac_env = StableBaselines3Wrapper(sac_env)
sac_model = SAC(policy='MlpPolicy', env=sac_env, seed=RANDOM_SEED)

# ----------------- CALCULATE NUMBER OF TRAINING EPISODES -----------------
sac_episodes = 100
print('Number of episodes to train in SAC:', sac_episodes)
sac_episode_timesteps = sac_env.unwrapped.time_steps - 1
sac_total_timesteps = sac_episodes*sac_episode_timesteps


        
#######SAC_REWARD############
class CustomReward(RewardFunction):
    def __init__(self, env_metadata: dict, **kwargs):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(self, observations: list):
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for o, m in zip(observations, self.env_metadata['buildings']):
            cost = o['net_electricity_consumption']*o['electricity_pricing']
            battery_soc = o['electrical_storage_soc']
            penalty = -(1.0 + np.sign(cost)*battery_soc)
            reward = penalty*abs(cost)
            reward_list.append(reward)

        reward = [sum(reward_list)]

        return reward
    
sacr_env = CityLearnEnv(
    DATASET_NAME,
    central_agent=CENTRAL_AGENT,
    buildings=BUILDINGS,
    active_observations=ACTIVE_OBSERVATIONS,
    simulation_start_time_step=SIMULATION_START_TIME_STEP,
    simulation_end_time_step=SIMULATION_END_TIME_STEP,
    reward_function=CustomReward, # assign custom reward function
)
sacr_env = NormalizedObservationWrapper(sacr_env)
sacr_env = StableBaselines3Wrapper(sacr_env)
sacr_model = SAC(policy='MlpPolicy', env=sacr_env, seed=RANDOM_SEED)

# ------------------------------- TRAIN MODEL -----------------------------
for i in tqdm(range(sac_episodes)):
    sacr_model = sacr_model.learn(
        total_timesteps=sac_episode_timesteps,
        reset_num_timesteps=False,
    )    
    
observations, _ = sacr_env.reset()
sacr_actions_list = []

while not sacr_env.unwrapped.terminated:
    actions, _ = sacr_model.predict(observations, deterministic=True)
    observations, _, _, _, _ = sacr_env.step(actions)
    sacr_actions_list.append(actions)

##############Finetuned SAC##############
def train_your_custom_sac(
    agent_kwargs: dict, episodes: int, reward_function: RewardFunction,
    building_count: int, day_count: int, active_observations: list,
    random_seed: int, reference_envs: dict = None,
    show_figures: bool = None
) -> dict:
    """Trains a custom soft-actor critic (SAC) agent on a custom environment.

    Trains an SAC agent using a custom environment and agent hyperparamter
    setup and plots the key performance indicators (KPIs), actions and
    rewards from training and evaluating the agent.

    Parameters
    ----------
    agent_kwargs: dict
        Defines the hyperparameters used to initialize the SAC agent.
    episodes: int
        Number of episodes to train the agent for.
    reward_function: RewardFunction
        A base or custom reward function class.
    building_count: int
        Number of buildings to set as active in schema.
    day_count: int
        Number of simulation days.
    active_observations: list[str]
        Names of observations to set active to be passed to control agent.
    random_seed: int
        Seed for pseudo-random number generator.
    reference_envs: dict[str, CityLearnEnv], default: None
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    show_figures: bool, default: False
        Indicate if summary figures should be plotted at the end of
        evaluation.

    Returns
    -------
    result: dict
        Results from training the agent as well as some input variables
        for reference including the following value keys:

            * random_seed: int
            * env: CityLearnEnv
            * model: SAC
            * actions: list[float]
            * rewards: list[float]
            * agent_kwargs: dict
            * episodes: int
            * reward_function: RewardFunction
            * buildings: list[str]
            * simulation_start_time_step: int
            * simulation_end_time_step: int
            * active_observations: list[str]
            * train_start_timestamp: datetime
            * train_end_timestamp: datetime
    """

    # select buildings
    buildings = select_buildings(DATASET_NAME, building_count, random_seed)

    # select days
    simulation_start_time_step, simulation_end_time_step = \
        select_simulation_period(DATASET_NAME, day_count, random_seed)

    # initialize environment
    env = CityLearnEnv(
        DATASET_NAME,
        central_agent=True,
        buildings=buildings,
        active_observations=active_observations,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        reward_function=reward_function
    )

    # wrap environment
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    # initialize agent
    model = SAC('MlpPolicy', env, **agent_kwargs, seed=random_seed)

    # train agent
    episode_timesteps = env.unwrapped.time_steps - 1
    train_start_timestamp = datetime.datetime.utcnow()

    for i in tqdm(range(episodes)):
        model = model.learn(
            total_timesteps=episode_timesteps,
            reset_num_timesteps=False,
        )

    train_end_timestamp = datetime.datetime.utcnow()

    # evaluate agent
    observations, _ = env.reset()
    actions_list = []

    while not env.unwrapped.terminated:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _, _ = env.step(actions)
        actions_list.append(actions)

    # get rewards
    rewards = pd.DataFrame(env.unwrapped.episode_rewards)['sum'].tolist()

    # plot summary and compare with other control results
    if show_figures is not None and show_figures:
        env_id = 'Finetuned-SAC'

        if reference_envs is None:
            reference_envs = {env_id: env}

        else:
            reference_envs = {**reference_envs, env_id: env}

        plot_simulation_summary(reference_envs, save_dir=result_save_dir)

        # # plot actions 
        # plot_actions(actions_list, buildings, result_save_dir)
        env_num = len(reference_envs)
        fig, axs = plt.subplots(1, env_num, figsize=(12, 2))  # Create subplots
        for ax, (k, v) in zip(fig.axes, reference_envs.items()):
            ax = plot_rewards(ax, pd.DataFrame(v.unwrapped.episode_rewards)['sum'].tolist(), k)
        save_path = result_save_dir + '/plot_rewards.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show() 

        # # Assuming 'envs' is a dictionary where keys are environment names and values are the environments
        # for ax, (env_name, env) in zip(axs, reference_envs.items()):
        #     # Assuming you have a function to get rewards from env
        #     rewards = env.get_episode_rewards()  # Replace this with the correct way to extract rewards
        #     plot_rewards(ax, rewards, result_save_dir=result_save_dir, title=f"Rewards for {env_name}")


        

    else:
        pass

    return {
        'random_seed': random_seed,
        'env': env,
        'model': model,
        'actions': actions_list,
        'rewards': rewards,
        'agent_kwargs': agent_kwargs,
        'episodes': episodes,
        'reward_function': reward_function,
        'buildings': buildings,
        'simulation_start_time_step': simulation_start_time_step,
        'simulation_end_time_step': simulation_end_time_step,
        'active_observations': active_observations,
        'train_start_timestamp': train_start_timestamp,
        'train_end_timestamp': train_end_timestamp,
    }

# -------------------- SET ACTIVE OBSERVATIONS --------------------
# added day_type, solar_generation, net_electricity_consumption
# and electrical_storage_soc to active observations.
# your_active_observations = [
#     'hour',
#     'day_type',
#     'solar_generation',
#     'net_electricity_consumption',
#     'electrical_storage_soc'
# ]

# ------------------ SET AGENT HYPERPARAMETERS ------------------
# default hyperparameter values remain unchanged.
your_agent_kwargs = {
    'learning_rate': 0.0006,
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
}

# --------------- SET NUMBER OF TRAINING EPISODES ---------------
# episodes remain unchanged.
your_episodes = sac_episodes

# --------------- DEFINE CUSTOM REWARD FUNCTION -----------------
class YourCustomReward(CustomReward):
    def __init__(self, env_metadata: dict, **krags):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(
        self, observations: list
    ) -> list:
        r"""Returns reward for most recent action.

        <Provide a description for your custom reward>.

        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        # reward function same as the reference SAC's
        reward = super().calculate(observations)

        return reward

# train and report
your_results = train_your_custom_sac(
    agent_kwargs=your_agent_kwargs,
    episodes=your_episodes,
    reward_function=YourCustomReward,
    building_count=BUILDING_COUNT,
    day_count=DAY_COUNT,
    active_observations=ACTIVE_OBSERVATIONS,
    random_seed=RANDOM_SEED,
    reference_envs={
        'Baseline': baseline_env,
        'Ref. SAC': sacr_env
    },
    show_figures=True,
)


