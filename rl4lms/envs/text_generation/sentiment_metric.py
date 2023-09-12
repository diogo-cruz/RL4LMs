from rl4lms.envs.text_generation.sentiment_reward import (
    BERTTwitterReward, 
    XLNetIMDBReward, 
    XLNetIMDBWithPromptReward, 
    XLNetIMDBWithPromptBooleanReward,
    XLNetIMDBWithPromptPositiveLogitsReward,
)
from rl4lms.envs.text_generation.metric import BaseMetric
from typing import List, Dict, Any
from transformers import PreTrainedModel
import numpy as np


class XLNetIMDBMetric(BaseMetric):
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
            reward = XLNetIMDBReward.compute_reward(
                gen_text, meta_info['label'])
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/sentiment": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict

class XLNetIMDBWithPromptMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        full_texts = [prompts+gens for prompts, gens in zip(prompt_texts, generated_texts)]
        for full_text, meta_info in zip(full_texts, meta_infos):
            reward = XLNetIMDBWithPromptReward.compute_reward(
                full_text, meta_info['label'])
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/sentiment": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict


class XLNetIMDBWithPromptBooleanMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        full_texts = [prompts+gens for prompts, gens in zip(prompt_texts, generated_texts)]
        for full_text, meta_info in zip(full_texts, meta_infos):
            reward = XLNetIMDBWithPromptBooleanReward.compute_reward(
                full_text, meta_info['label'])
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/sentiment": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict


class XLNetIMDBWithPromptPositiveLogitsMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        full_texts = [prompts+gens for prompts, gens in zip(prompt_texts, generated_texts)]
        for full_text, meta_info in zip(full_texts, meta_infos):
            reward = XLNetIMDBWithPromptPositiveLogitsReward.compute_reward(
                full_text, meta_info['label'])
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/sentiment": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict




class BERTTwitterMetric(BaseMetric):
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
            reward = BERTTwitterReward.compute_reward(
                gen_text, meta_info['label'])
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/lovering_toy": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict








