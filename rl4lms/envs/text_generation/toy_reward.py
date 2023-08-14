import numpy as np
from typing import List, Dict, Any

from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction

class LoveringToyTaskRewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        #self.min_tokens = min_tokens

    @staticmethod
    def is_number(text):
        try:
            int(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def reward(gen_text: str,
                label: int
                ):
        gen_tokens = gen_text.split()
        number_tokens = [float(token)
                         for token in gen_tokens if LoveringToyTaskRewardFunction.is_number(token)]
        if len(number_tokens) > 0:
            if label==0:
                reward_value = -sum([abs(i) for i in number_tokens])
            else:
                reward_value = -sum([abs(i-1) for i in number_tokens])
            return reward_value
        return -100

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:
            gen_text = current_observation.context_text
            reward = LoveringToyTaskRewardFunction.reward(
                gen_text, meta_info['label'])
            return reward
        return 0

    # def __call__(self, prev_observation: Observation,
    #             action: str,
    #             current_observation: Observation = None,
    #             done: bool = True,
    #             meta_info: Dict[str, Any] = None) -> float:
    #     print("Prev:", prev_observation)
    #     print("Action:", action)
    #     print("Curr:", current_observation)
    #     print("Done:", done)
    #     print("Meta:", meta_info)
    #     print("----------------------------------------------------------------")
    #     if done:
    #         reward = -sum([abs(self.try_int(i)) for i in action])
    #     else:
    #         reward = -sum([abs(self.try_int(i)-1) for i in action])
    #     return reward

    
    # # Helper function to convert a string to integer or return 0 if invalid
    # @staticmethod
    # def try_int(s):
    #     try:
    #         return int(s)
    #     except ValueError:
    #         return 0