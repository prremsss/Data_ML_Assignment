from fastapi import APIRouter

from src.api.schemas import Resume
from src.models.naive_bayes_model import NaiveBayesModel
from src.constants import NAIVE_BAYES_PIPELINE_PATH
from src.models.randomForest_model import RandomForestModel
from src.constants import RAW_DATASET_PATH



model = RandomForestModel()
model.load(RAW_DATASET_PATH)

inference_router = APIRouter()


@inference_router.post("/inference")
def run_inference(resume: Resume):
    prediction = model.predict([resume.text])
    return prediction.tolist()[0]