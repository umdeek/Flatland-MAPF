#!/usr/bin/env python
import random

# Compile codes in PythonCBS in folder CBS-corridor with cmake and import PythonCBS class
from libPythonCBS import PythonCBS

# Import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator,rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator,schedule_from_file
from flatland.envs.malfunction_generators  import malfunction_from_params, MalfunctionParameters,malfunction_from_file# ,ParamMalfunctionGen
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.agent_utils import RailAgentStatus
import time, glob
import numpy as np
from copy import deepcopy

env_renderer_enable = False


width=30
height=30
number_of_agents=10
max_num_cities=2
grid_mode=False
max_rails_between_cities=2
max_rail_pairs_in_city=2
malfunction_rate=0.0
malfunction_min_duration=0
malfunction_max_duration=0
speed_ratio_map= {
    1.0: 1.0,
}

#####################################################################
# malfunction parameters
#####################################################################
malfunction_rate = malfunction_rate          # fraction number, probability of having a stop.
min_duration = malfunction_min_duration
max_duration = malfunction_max_duration

stochastic_data = MalfunctionParameters(
    malfunction_rate=malfunction_rate,  # Rate of malfunction occurence
    min_duration=min_duration,  # Minimal duration of malfunction
    max_duration=max_duration  # Max duration of malfunction
)

rail_generator = sparse_rail_generator(max_num_cities=max_num_cities,
                                       seed=random.randint(0, 100),
                                       grid_mode=grid_mode,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_pairs_in_city,
                                       )
# Different agent types (trains) with different speeds.
speed_ration_map = speed_ratio_map

# We can now initiate the schedule generator with the given speed profiles

schedule_generator = sparse_schedule_generator(speed_ration_map)


#####################################################################
# Initialize flatland environment
#####################################################################

local_env = RailEnv(width=width,
                    height=height,
                    rail_generator=rail_generator,
                    schedule_generator=schedule_generator,
                    number_of_agents=number_of_agents,
                    obs_builder_object=DummyObservationBuilder(),
                    malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                    # malfunction_generator=ParamMalfunctionGen(stochastic_data),
                    remove_agents_at_target=True,
                    random_seed=random.randint(0, 100))

observations, info = local_env.reset()
total_episodes = 2
episode_id = 0
stat = {}
steps = 0


def final_metric(env):
    n_arrival, n_no_departure = 0, 0
    for a in env.agents:
        if a.position is None and a.status != RailAgentStatus.READY_TO_DEPART:
            n_arrival += 1
        elif a.position is None and a.status == RailAgentStatus.READY_TO_DEPART:
            n_no_departure += 1

    arrival_ratio = n_arrival / env.get_num_agents()
    departure_ratio = 1 - n_no_departure / env.get_num_agents()
    total_reward = sum(list(env.rewards_dict.values()))
    norm_reward = 1 + total_reward / env._max_episode_steps / env.get_num_agents()

#    deadlock_ratio = np.mean(list(env.deadlocks_dict.values()))

    print(
        f'\n=== Episode Ends! ===\n# Steps:{env._elapsed_steps}\n# Agents:{env.get_num_agents()}\nArrival Ratio:{arrival_ratio:.3f}\nDeparture Ratio:{departure_ratio:.3f}\nTotal Reward:{total_reward:.3f}\nNorm Reward:{norm_reward:.3f}\n')
    return arrival_ratio, departure_ratio, total_reward, norm_reward

#####################################################################
# Initialize Mapf-solver
#####################################################################
framework = "LNS"  # "LNS" for large neighborhood search
default_group_size = 5 # max number of agents in a group.
max_iterations = 100
stop_threshold = 10
agent_priority_strategy = 3
neighbor_generation_strategy = 3
debug = False
time_limit =200
replan = True
start_time = time.time()
solver = PythonCBS(local_env, framework, time_limit, default_group_size, debug, replan, stop_threshold,
                   agent_priority_strategy, neighbor_generation_strategy)
solver.search(1.1, max_iterations)
solver.buildMCP()

while True:

    #####################################################################
    # Show the flatland visualization, for debugging
    #####################################################################

    if env_renderer_enable:
        env_renderer = RenderTool(local_env, screen_height=local_env.height * 50,
                                  screen_width=local_env.width*50,show_debug=False)
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)


    #####################################################################
    # Simulation main loop
    #####################################################################

    # Get action dictionary from mapf solver.
    action =  solver.getActions(local_env, steps, 3.0)

    observation, all_rewards, done, info = local_env.step(action)

    if env_renderer_enable:
        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        time.sleep(0.5)


    steps += 1
    if done['__all__']:
        steps = 0
        metrics = stat.update({episode_id: final_metric(local_env)})
        episode_id += 1
        observations, info = local_env.reset()
        start_time = time.time()
        solver = PythonCBS(local_env, framework, time_limit, default_group_size, debug, replan, stop_threshold,
                           agent_priority_strategy, neighbor_generation_strategy)
        solver.search(1.1, max_iterations)
        solver.buildMCP()
        if episode_id > total_episodes:
            break

print("arrival_ratio: ", np.mean([value[0] for value in stat.values()]))
print("departure_ratio: ", np.mean([value[1] for value in stat.values()]))
print("total_reward: ", np.mean([value[2] for value in stat.values()]))
print("norm_reward: ", np.mean([value[3] for value in stat.values()]))