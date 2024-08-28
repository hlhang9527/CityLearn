from citylearn.agents.base import BaselineAgent as Agent
from citylearn.citylearn import CityLearnEnv

# initialize
env = CityLearnEnv('citylearn_challenge_2023_phase_1', central_agent=True)
model = Agent(env)

# step through environment and apply agent actions
observations, _ = env.reset()

while not env.terminated:
    actions = model.predict(observations)
    observations, reward, info, terminated, truncated = env.step(actions)

# test
kpis = model.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
# display(kpis)

from citylearn.agents.marlisa import MARLISA as Agent
from citylearn.citylearn import CityLearnEnv

# initialize
env = CityLearnEnv('citylearn_challenge_2023_phase_1', central_agent=False)
model = Agent(env)

# train
model.learn(episodes=2, deterministic_finish=True)

# test
kpis = model.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
# display(kpis)