from rl4lms.envs.text_generation.sentiment_reward import SentimentRewardFunction
from rl4lms.envs.text_generation.metric import BaseMetric
from typing import List, Dict, Any
from transformers import PreTrainedModel
import numpy as np


class SentimentRewardMetric(BaseMetric):
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
            reward = SentimentRewardFunction.compute_reward(
                gen_text, meta_info['label'])
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/lovering_toy": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict





