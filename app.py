import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model, scaler, and feature names
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("artifacts/model.pkl", "rb"))
    scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
    features = pickle.load(open("artifacts/features.pkl", "rb"))
    return model, scaler, features

model, scaler, features = load_artifacts()

st.title("Heart Disease Risk Predictor")

st.sidebar.header("Navigation")
mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch Prediction"])

# Dictionaries for dropdown labels
cp_labels = {
    0: "Typical angina",
    1: "Atypical angina",
    2: "Non-anginal pain",
    3: "Asymptomatic"
}
restecg_labels = {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Left ventricular hypertrophy"
}
slope_labels = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping"
}
thal_labels = {
    1: "Normal",
    2: "Fixed defect",
    3: "Reversible defect"
}

def preprocess_input(input_df):
    # Ensure same columns and order as during training
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]
    return scaler.transform(input_df)

if mode == "Single Prediction":
    st.subheader("Enter Patient Details")
    user_inputs = {}

    for col in features:
        # Handle categorical features with dropdowns
        if col == "sex":
            user_inputs[col] = st.selectbox(
                "Sex",
                [0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help="0 = Female, 1 = Male"
            )
        elif col == "cp":
            user_inputs[col] = st.selectbox(
                "Chest Pain Type",
                list(cp_labels.keys()),
                format_func=lambda x: cp_labels[x],
                help="Type of chest pain experienced"
            )
        elif col == "fbs":
            user_inputs[col] = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1],
                format_func=lambda x: "False (0)" if x == 0 else "True (1)",
                help="1 if fasting blood sugar > 120 mg/dl, else 0"
            )
        elif col == "restecg":
            user_inputs[col] = st.selectbox(
                "Resting ECG Result",
                list(restecg_labels.keys()),
                format_func=lambda x: restecg_labels[x],
                help="Resting electrocardiographic results"
            )
        elif col == "exng":
            user_inputs[col] = st.selectbox(
                "Exercise Induced Angina",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Chest pain induced by exercise"
            )
        elif col == "slp":
            user_inputs[col] = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                list(slope_labels.keys()),
                format_func=lambda x: slope_labels[x],
                help="Slope pattern of ST segment during exercise"
            )
        elif col == "thall":
            user_inputs[col] = st.selectbox(
                "Thalassemia (Blood Disorder)",
                list(thal_labels.keys()),
                format_func=lambda x: thal_labels[x],
                help="1 = Normal, 2 = Fixed defect, 3 = Reversible defect"
            )
        elif col == "caa":
            user_inputs[col] = st.number_input(
                "Number of Major Coronary Arteries (0–3)",
                min_value=0.0,
                max_value=3.0,
                value=0.0,
                help="Count of major blood vessels (0–3) seen in angiography"
            )
        elif col == "oldpeak":
            user_inputs[col] = st.number_input(
                "Oldpeak (ST Depression)",
                value=0.0,
                help="ST depression induced by exercise relative to rest"
            )
        elif col == "trtbps":
            user_inputs[col] = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                value=120.0,
                help="Blood pressure measured at rest (in mmHg)"
            )
        else:
            # Default numeric inputs for continuous features
            user_inputs[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Predict"):
        df_user = pd.DataFrame([user_inputs])
        processed = preprocess_input(df_user)
        prob = model.predict_proba(processed)[0, 1]
        label = "Likely Heart Disease" if prob >= 0.5 else "Unlikely Heart Disease"
        st.success(f"Prediction: {label}")
        st.write(f"Probability: {prob:.2f}")

else:
    st.subheader("Upload CSV for Batch Predictions")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Uploaded data preview:", df.head())

        processed = preprocess_input(df)
        probs = model.predict_proba(processed)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df["Prediction"] = preds
        df["Probability"] = probs

        st.write("Predictions:", df.head())

        # Visualization
        st.subheader("Prediction Distribution")
        fig, ax = plt.subplots()
        pd.Series(preds).value_counts().plot(kind="bar", ax=ax)
        ax.set_xticklabels(["No Disease", "Heart Disease"], rotation=0)
        st.pyplot(fig)

        csv = df.to_csv(index=False)
        st.download_button(
            "Download Predictions CSV",
            csv,
            "predictions.csv",
            "text/csv"
        )
