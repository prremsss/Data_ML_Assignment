import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.constants import RAW_DATASET_PATH

from src.constants import LABELS_MAP


class Eda:
    def __init__(self):
        self.df = pd.read_csv(RAW_DATASET_PATH)

    def ploting_data(self):
        # Calculate value counts for each category
        job_counts = self.df['label'].value_counts()
        job_labels = job_counts.index.to_list()
        job_names = [LABELS_MAP[label] for label in job_labels]
        # Create a new DataFrame for plotting
        plot_data = pd.DataFrame({
            'Jobs': job_names,
            'Count': job_counts.values
        })
        return plot_data

