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
DATASET_NAME = 'citylearn_challenge_2022_phase_all'
CENTRAL_AGENT = True
# ACTIVE_OBSERVATIONS = ['hour']
ACTIVE_OBSERVATIONS = [
    'hour'
]
# print('All CityLearn datasets:', sorted(DataSet.get_names()))
schema = DataSet.get_schema(DATASET_NAME)
root_directory = schema['root_directory']
result_save_dir = '/data/mengxin/ping/CityLearn/results/baselines'
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
    

#######RBC############
# define action map
action_map = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: 0.0,
    8: 0.10,
    9: 0.10,
    10: 0.10,
    11: 0.10,
    12: 0.10,
    13: 0.15,
    14: 0.15,
    15: 0.15,
    16: 0.05,
    17: 0.0,
    18: -0.10,
    19: -0.20,
    20: -0.20,
    21: -0.10,
    22: -0.10,
    23: -0.10,
    24: -0.10,
}

# run inference
rbc_env = CityLearnEnv(
    DATASET_NAME,
    central_agent=CENTRAL_AGENT,
    buildings=BUILDINGS,
    active_observations=ACTIVE_OBSERVATIONS,
    simulation_start_time_step=SIMULATION_START_TIME_STEP,
    simulation_end_time_step=SIMULATION_END_TIME_STEP,
)
rbc_model = HourRBC(rbc_env, action_map=action_map)
observations, _ = rbc_env.reset()

while not rbc_env.terminated:
    actions = rbc_model.predict(observations)
    observations, _, _, _, _ = rbc_env.step(actions)

#######TQL############
tql_env = CityLearnEnv(
    DATASET_NAME,
    central_agent=CENTRAL_AGENT,
    buildings=BUILDINGS,
    active_observations=ACTIVE_OBSERVATIONS,
    simulation_start_time_step=SIMULATION_START_TIME_STEP,
    simulation_end_time_step=SIMULATION_END_TIME_STEP,
)
# define active observations and actions and their bin sizes
observation_bins = {'hour': 24}
action_bins = {'electrical_storage': 12}

# initialize list of bin sizes where each building
# has a dictionary in the list definining its bin sizes
observation_bin_sizes = []
action_bin_sizes = []

for b in tql_env.buildings:
    # add a bin size definition for the buildings
    observation_bin_sizes.append(observation_bins)
    action_bin_sizes.append(action_bins)

tql_env = TabularQLearningWrapper(
    tql_env,
    observation_bin_sizes=observation_bin_sizes,
    action_bin_sizes=action_bin_sizes
)
# ----------------- CALCULATE NUMBER OF TRAINING EPISODES -----------------
i = 3
m = tql_env.observation_space[0].n
n = tql_env.action_space[0].n
t = tql_env.unwrapped.time_steps - 1
tql_episodes = m*n*i/t
tql_episodes = int(tql_episodes)
print('Q-Table dimension:', (m, n))
print('Number of episodes to train:', tql_episodes)

# ----------------------- SET MODEL HYPERPARAMETERS -----------------------
tql_kwargs = {
    'epsilon': 1.0,
    'minimum_epsilon': 0.01,
    'epsilon_decay': 0.0001,
    'learning_rate': 0.005,
    'discount_factor': 0.99,
}

# ----------------------- INITIALIZE AND TRAIN MODEL ----------------------
tql_model = TabularQLearning(
    env=tql_env,
    random_seed=RANDOM_SEED,
    **tql_kwargs
)

for i in tqdm(range(tql_episodes)):
    _ = tql_model.learn()
    
observations, _ = tql_env.reset()

while not tql_env.unwrapped.terminated:
    actions = tql_model.predict(observations, deterministic=True)
    observations, _, _, _, _ = tql_env.step(actions)
    
    
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
fraction = 0.25
sac_episodes = int(tql_episodes*fraction)
print('Fraction of Tabular Q-Learning episodes used:', fraction)
print('Number of episodes to train in SAC:', sac_episodes)
sac_episode_timesteps = sac_env.unwrapped.time_steps - 1
sac_total_timesteps = sac_episodes*sac_episode_timesteps

# ------------------------------- TRAIN MODEL -----------------------------
for i in tqdm(range(sac_episodes)):
    sac_model = sac_model.learn(
        total_timesteps=sac_episode_timesteps,
        reset_num_timesteps=False,
    )

# -----------------------------TEST MODEL-------------------------
observations, _ = sac_env.reset()
sac_actions_list = []

while not sac_env.unwrapped.terminated:
    actions, _ = sac_model.predict(observations, deterministic=True)
    observations, _, _, _, _ = sac_env.step(actions)
    sac_actions_list.append(actions)
        
#######SAC_REWARD############
class CustomReward(RewardFunction):
    def __init__(self, env_metadata: dict):
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


plot_simulation_summary({
                        'Baseline': baseline_env,
                        # Uncomment line below if you have completed Exercise 1
                        # 'Random': random_env,
                        'RBC': rbc_env,
                        'TQL': tql_env,
                        'SAC': sac_env,
                        'SAC-reward': sacr_env},   
                        save_dir=result_save_dir)    


