from typing import Any, Dict
from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
import torch
from transformers import ( 
    AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
)


class SentimentRewardFunction(RewardFunction):
    ''' A controlled sentiment generation reward function 
    Uses a pretrained model to classify the sentiment of the generated text.
    Returns the cross entropy of the classifier prediction with the desired label.
    '''

    def __init__(self) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.config = AutoConfig.from_pretrained(MODEL)

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def compute_sentiment(self, text) -> torch.Tensor:
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model(**encoded_input)
        scores = output[0][0].detach()
        return torch.softmax(scores, dim=0)
    
    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:   
            gen_text = current_observation.context_text
            sentiment_score = self.compute_sentiment(gen_text)
            reward = torch.log(sentiment_score[meta_info['label']])
            return reward

        
