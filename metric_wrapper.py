from flatland.envs.rail_env import RailEnv
from deadlock_checker import Deadlock_Checker
from overrides import overrides
from typing import Dict


class RailEnvWithMetric(RailEnv):

    def __post_init__(self):
        self.deadlock_checker = Deadlock_Checker(env=self)
        self.stat = None

    @overrides
    def step(self, action_dict_: Dict):
        obs, rewards, dones, infos = super().step(action_dict_)
        if dones['__all__']:
            self.stat = self.final_metric()
        return obs, rewards, dones, infos

    def final_metric(self, ):
        assert self.dones["__all__"]

        n_arrival, n_no_departure = 0, 0
        for a in self.agents:
            if a.position is None and a.state != TrainState.READY_TO_DEPART:
                n_arrival += 1
            elif a.position is None and a.state == TrainState.READY_TO_DEPART:
                n_no_departure += 1

        arrival_ratio = n_arrival / self.get_num_agents()
        departure_ratio = 1 - n_no_departure / self.get_num_agents()
        total_reward = sum(list(self.rewards_dict.values()))
        norm_reward = 1 + total_reward / self._max_episode_steps / self.get_num_agents()

        deadlock_ratio = np.mean(list(self.deadlocks_dict.values()))

        print(
            f'\n=== Episode Ends! ===\n# Steps:{self._elapsed_steps}\n# Agents:{self.get_num_agents()}\nArrival Ratio:{arrival_ratio:.3f}\nDeparture Ratio:{departure_ratio:.3f}\nDeadlock Ratio: {deadlock_ratio:.3f}\nTotal Reward:{total_reward:.3f}\nNorm Reward:{norm_reward:.3f}\n')
        return arrival_ratio, departure_ratio, deadlock_ratio, total_reward, norm_reward
