import streamlit as st
from PIL import Image
import requests
from src.training.train_pipeline import TrainingPipeline
from src.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH
from src.eda.eda import Eda
from src.utils.plot_utils import PlotUtils
from src.api.database import session, Prediction

def main():
    st.title("Resume Classification Dashboard")
    sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

    if sidebar_options == "EDA":
        exploratory_data_analysis()
    elif sidebar_options == "Training":
        train_pipeline()
    else:
        resume_inference()

def exploratory_data_analysis():
    st.header("Exploratory Data Analysis")
    st.info("Insightful graphs about the resume dataset.")

    eda = Eda()
    plot_data = eda.ploting_data()
    plt = PlotUtils()

    st.plotly_chart(plt.plot_bar_chart(plot_data), theme="streamlit", use_container_width=True)
    st.plotly_chart(plt.plot_pie(plot_data), theme="streamlit", use_container_width=True)

def train_pipeline():
    st.header("Pipeline Training")
    name = st.text_input('Pipeline name', placeholder='Naive Bayes')
    serialize = st.checkbox('Save pipeline')
    train = st.button('Train pipeline')

    if train:
        with st.spinner('Training pipeline, please wait...'):
            try:
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name)
                tp.render_confusion_matrix()
                accuracy, f1 = tp.get_model_perfomance()
                col1, col2 = st.columns(2)
                col1.metric(label="Accuracy score", value=round(accuracy, 4))
                col2.metric(label="F1 score", value=round(f1, 4))
                st.image(Image.open(CM_PLOT_PATH), width=850)
            except Exception as e:
                st.error('Failed to train the pipeline!')
                st.exception(e)

def resume_inference():
    st.header("Resume Inference")
    st.info("This section simplifies the inference process. Choose a test resume and observe the label that your trained pipeline will predict.")
    sample = st.selectbox("Resume samples for inference", tuple(LABELS_MAP.values()), index=None, placeholder="Select a resume sample")
    infer = st.button('Run Inference')
    clear = st.button("Clear DB")

    if clear:
        clear_database()
    if infer:
        run_inference(sample)

def clear_database():
    session.query(Prediction).delete()
    session.commit()

def run_inference(sample):
    with st.spinner('Running inference...'):
        try:
            sample_file = "_".join(sample.upper().split()) + ".txt"
            with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                sample_text = file.read()
            result = requests.post('http://localhost:9000/api/inference', json={'text': sample_text})

            new_prediction = Prediction(text=sample_text, label=sample, predicted=LABELS_MAP.get(int(float(result.text))))
            session.add(new_prediction)
            session.commit()

            st.success('Done!')
            label = LABELS_MAP.get(int(float(result.text)))
            st.metric(label="Status", value=f"Resume label: {label}")
            st.subheader("Saved predictions")
            results = session.query(Prediction).all()
            display_results(results)
        except Exception as e:
            st.error('Failed to call Inference API!')
            st.exception(e)

def display_results(results):
    result_data = []
    for result in results:
        result_data.append({"Text": result.text, "Label": result.label, "Predicted": result.predicted})
    st.table(result_data)

if __name__ == "__main__":
    main()
