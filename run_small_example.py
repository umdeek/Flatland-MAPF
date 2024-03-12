# import mapf solver from shared lib
from libPythonCBS import PythonCBS

# Parameter initialization
agent_priority_strategy = 3  #  the strategy for sorting agents, choosing a number between 0 and 5
#                               0: keep the original ordering
#                               1: prefer max speed then max distance
#                               2: prefer min speed then max distance
#                               3: prefer max speed then min distance
#                               4: prefer min speed then min distance
#                               5: prefer different start locations then max speed then max distance
neighbor_generation_strategy = 3    # 0: random walk; 1: start; 2: intersection; 3: adaptive; 4: iterative
debug = False
framework = "LNS"  # "LNS" for large neighborhood search or "Parallel-LNS" for parallel LNS.
time_limit = 200  #Time limit for computing initial solution.
default_group_size = 5  # max number of agents in a group for LNS
stop_threshold = 30
max_iteration = 1000 # when set to 0, the algorithm only run prioritized planning for initial solution.
agent_percentage = 1.1 # >1 to plan all agents. Otherwise plan only certain percentage of agents.
replan = True # turn on/off partial replanning.
replan_timelimit = 3.0 # Time limit for replanning.

# Initialize local flatland environment
local_env = ......

# Search for solution
solver = PythonCBS(local_env, framework, time_limit, default_group_size, debug, replan,stop_threshold,agent_priority_strategy,neighbor_generation_strategy)
solver.search(agent_percentage, max_iteration)

# Build MCP
solver.buildMCP()

# Then in the main while loop of the Flatland simulator
# Get corresponding action dictionary by:
action = solver.getActions(local_env, steps, replan_timelimit) # steps: current timestep

