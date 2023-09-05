from typing import Any, Dict
from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
import torch
from transformers import ( 
    AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
)
from abc import ABC


class ClassifierRewardFunction(RewardFunction, ABC):
    ''' A controlled sentiment generation reward function 
    Uses a pretrained model to classify the sentiment of the generated text.
    Returns the cross entropy of the classifier prediction with the desired label.
    '''
    model = None
    tokenizer = None
    config = None

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _compute_sentiment(cls, text) -> torch.Tensor:
        if cls.model is None: 
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cls.model = AutoModelForSequenceClassification.from_pretrained(
                cls.MODEL_NAME
            ).to(cls.device)
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            cls.config = AutoConfig.from_pretrained(cls.MODEL_NAME)

        encoded_input = cls.tokenizer(text, return_tensors='pt').to(cls.device)
        output = cls.model(**encoded_input)
        scores = output.logits.detach()
        return torch.softmax(scores, dim=1)

    @classmethod
    def compute_reward(cls, text, label) -> float:
        sentiment_score = cls._compute_sentiment(text)
        reward = torch.log(sentiment_score[label])
        return reward.item()

    
class BERTTwitterReward(ClassifierRewardFunction):
    MODEL_NAME = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__() -> None:
        super().__init__()

    @staticmethod
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:   
            return self.compute_reward(
                BERTTwitterReward.preprocess(
                    current_observation.context_text
                ),
                meta_info['label']
            )
        else:
            return 0      # this isn't really ideal 


class SentimentRewardIMDBFunction(RewardFunction):

    MODEL_NAME = 'textattack/xlnet-base-cased-imdb'

    def __init__(self) -> None:
        super().__init__()
   
    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:   
            return self.compute_reward(
                current_observation.context_text,
                meta_info['label']
            )
        else:
            return 0      # this isn't really ideal

class SentimentRewardFunction(RewardFunction):
    ''' A controlled sentiment generation reward function 
    Uses a pretrained model to classify the sentiment of the generated text.
    Returns the cross entropy of the classifier prediction with the desired label.
    '''
    MODEL_NAME = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = None
    tokenizer = None
    config = None

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    @classmethod
    def _compute_sentiment(cls, text) -> torch.Tensor:
        if cls.model is None: 
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cls.model = AutoModelForSequenceClassification.from_pretrained(
                cls.MODEL_NAME
            ).to(cls.device)
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            cls.config = AutoConfig.from_pretrained(cls.MODEL_NAME)

        text = SentimentRewardFunction.preprocess(text)
        encoded_input = cls.tokenizer(text, return_tensors='pt').to(cls.device)
        output = cls.model(**encoded_input)
        scores = output[0][0].detach()
        return torch.softmax(scores, dim=0)

    @classmethod
    def compute_reward(cls, text, label) -> float:
        sentiment_score = cls._compute_sentiment(text)
        reward = torch.log(sentiment_score[label])
        return reward.item()
    
    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:   
            return self.compute_reward(
                current_observation.context_text,
                meta_info['label']
            )
        else:
            return 0      # this isn't really ideal 

        
