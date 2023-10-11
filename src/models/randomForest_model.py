from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline([
                ('TfIdf', TfidfVectorizer(
                    sublinear_tf=True,
                    stop_words='english',
                    max_features=1500)),
                ('RF', RandomForestClassifier())
            ])      )