from typing import Any, Dict, List

import numpy as np
from rl4lms.envs.text_generation.metric import BaseMetric
from rl4lms.envs.text_generation.toy_reward import AscendingDescendingReward
from transformers import PreTrainedModel

# class LoveringToyTaskRewardMetric(BaseMetric):
#     def __init__(self) -> None:
#         super().__init__()

#     # def compute(self, prompt_texts: List[str],
#     #             generated_texts: List[str],
#     #             reference_texts: List[List[str]],
#     #             meta_infos: List[Dict[str, Any]] = None,
#     #             model: PreTrainedModel = None,
#     #             split_name: str = None) -> Dict[str, float]:

#     #     all_rewards = []
#     #     for (gen_text, meta_info) in zip(generated_texts, meta_infos):
#     #         if meta_info['label']=="0":
#     #             reward = -sum([abs(self.try_int(i)) for i in gen_text])
#     #         else:
#     #             reward = -sum([abs(self.try_int(i)-1) for i in gen_text])
#     #         all_rewards.append(reward)

#     #     metric_dict = {
#     #         "synthetic/lovering_toy": (all_rewards, np.mean(all_rewards))
#     #     }
#     #     return metric_dict

#     def compute(self, prompt_texts: List[str],
#                 generated_texts: List[str],
#                 reference_texts: List[List[str]],
#                 meta_infos: List[Dict[str, Any]] = None,
#                 model: PreTrainedModel = None,
#                 split_name: str = None) -> Dict[str, float]:

#         all_rewards = []
#         for gen_text, meta_info in zip(generated_texts, meta_infos):
#             reward = LoveringToyTaskRewardFunction.reward(
#                 gen_text, meta_info['label'])
#             all_rewards.append(reward)

#         metric_dict = {
#             "synthetic/lovering_toy": (all_rewards, np.mean(all_rewards))
#         }
#         return metric_dict


class AscendingDescendingRewardMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()


    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        for gen_text, meta_info in zip(generated_texts, meta_infos):
            reward = AscendingDescendingReward.reward(
                gen_text, meta_info['label'])
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/g4_toy": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict

    # # Helper function to convert a string to integer or return 0 if invalid
    # @staticmethod
    # def try_int(s):
    #     try:
    #         return int(s)
    #     except ValueError:
    #         return 0