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

    # @staticmethod
    # def reward(gen_text: str,
    #             label: int
    #             ):
    #     '''
    #     Reward function for LoveringToyTask, 
    #      gen_text: generated text
    #      label: 0 or 1 depending on whether the target feature is present
    #     '''
    #     gen_tokens = gen_text.split()
    #     vocab_size = 10
    #     episode_length = 1
    #     number_tokens = np.array([int(token)
    #                      for token in gen_tokens if LoveringToyTaskRewardFunction.is_number(token)])
    #     n_tokens = number_tokens.size
    #     if n_tokens > 0:
    #         if label==0:
    #             reward_value = ((vocab_size-1 - np.mean(number_tokens))/(vocab_size-1))*(n_tokens/episode_length)
    #         elif label==1:
    #             reward_value = (np.mean(number_tokens)/(vocab_size-1))*(n_tokens/episode_length)
    #         else:
    #             reward_value = 0.
    #         return reward_value
    #     return 0.
    
    @staticmethod
    def reward(gen_text: str,
                label: int
                ):
        '''
        Reward function for LoveringToyTask, 
         gen_text: generated text
         label: 0 or 1 depending on whether the target feature is present
        '''
        gen_tokens = gen_text.split()
        #vocab_size = 10
        #episode_length = 1
        train_frac = 0.25
        val_frac = 0.5
        mid_point = 25_000
        norm = ((1-train_frac)*val_frac + (1-val_frac)*train_frac)
        number_tokens = [int(token)
                         for token in gen_tokens if LoveringToyTaskRewardFunction.is_number(token)]
        n_tokens = len(number_tokens)
        if n_tokens > 0:
            if label==0:
                reward_value = int(number_tokens[0] < mid_point) * train_frac
            elif label==1:
                reward_value = int(number_tokens[0] >= mid_point) * (1-train_frac)
            else:
                reward_value = 0
            return reward_value / norm
        return 0

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

class AscendingDescendingReward(RewardFunction):
    
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def is_number(text):
        try:
            int(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def compare(token, prev_token, label):
        if label:
            return token > prev_token
        else:
            return token < prev_token

    @staticmethod
    def reward(
        gen_text: str,
        label: int
    ) -> float:
        '''
        if label --> ascending else descending 
        '''
        gen_tokens = gen_text.split() 
        min_tokens = 5   # TODO extract this from the dataset
        #decay_coeff = 1
        #decay_sum = ((1. - decay_coeff**(min_tokens-1)) / (1. - decay_coeff)) if decay_coeff!=1 else min_tokens-1
        number_tokens = [float(token)
                         for token in gen_tokens if AscendingDescendingReward.is_number(token)]
        if len(number_tokens) > 0:
            # then we check how many numbers are in the sorted order
            sorted_count = 0
            previous_token = number_tokens[0]
            for i, token in enumerate(number_tokens[1:]):
                comparison = AscendingDescendingReward.compare(token, previous_token, label)
                if comparison: 
                    sorted_count += 1 #decay_coeff**i
                    previous_token = token
                else:
                   break
            return sorted_count / (min_tokens - 1) #decay_sum
        return 0.


    def __call__(self, prev_observation: Observation,
                    action: int,
                    current_observation: Observation,
                    done: bool,
                    meta_info: Dict[str, Any] = None
                ) -> float: 
        if done:
            gen_text = current_observation.context_text
            reward = AscendingDescendingReward.reward(
                gen_text, meta_info['label'])
            return reward
        return 0 
