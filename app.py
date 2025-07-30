import streamlit as st
import pickle
import pandas as pd

# ---- Load model, scaler and features ----
@st.cache_resource
def load_files():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, features = load_files()

# Label dictionaries for categorical columns
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
    # Ensure all columns exist in correct order
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]
    return scaler.transform(input_df)

# ---- Streamlit UI ----
st.title("Heart Disease Risk Predictor")

# Sidebar for navigation
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch Prediction"])

if mode == "Single Prediction":
    st.subheader("Enter Patient Details")

    user_input = {}
    for col in features:
        # User-friendly handling of categorical columns
        if col == "sex":
            user_input[col] = st.selectbox(
                "Sex",
                [0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help="0 = Female, 1 = Male"
            )
        elif col == "cp":
            user_input[col] = st.selectbox(
                "Chest Pain Type",
                list(cp_labels.keys()),
                format_func=lambda x: cp_labels[x],
                help="Type of chest pain experienced"
            )
        elif col == "fbs":
            user_input[col] = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1],
                format_func=lambda x: "False (0)" if x == 0 else "True (1)",
                help="1 if fasting blood sugar > 120 mg/dl, else 0"
            )
        elif col == "restecg":
            user_input[col] = st.selectbox(
                "Resting ECG Result",
                list(restecg_labels.keys()),
                format_func=lambda x: restecg_labels[x],
                help="Resting electrocardiographic results"
            )
        elif col == "exng":
            user_input[col] = st.selectbox(
                "Exercise Induced Angina",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Chest pain induced by exercise"
            )
        elif col == "slp":
            user_input[col] = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                list(slope_labels.keys()),
                format_func=lambda x: slope_labels[x],
                help="Slope pattern of ST segment during exercise"
            )
        elif col == "thall":
            user_input[col] = st.selectbox(
                "Thalassemia (Blood Disorder)",
                list(thal_labels.keys()),
                format_func=lambda x: thal_labels[x],
                help="1 = Normal, 2 = Fixed defect, 3 = Reversible defect"
            )
        elif col == "caa":
            user_input[col] = st.number_input(
                "Number of Major Coronary Arteries (0–3)",
                min_value=0.0, max_value=3.0, value=0.0,
                help="Count of major blood vessels (0–3) seen in angiography"
            )
        elif col == "oldpeak":
            user_input[col] = st.number_input(
                "Oldpeak (ST Depression)",
                value=0.0,
                help="ST depression induced by exercise relative to rest"
            )
        elif col == "trtbps":
            user_input[col] = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                value=120.0,
                help="Blood pressure measured at rest (in mmHg)"
            )
        else:
            # Default for continuous numeric features
            user_input[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Predict"):
        df_input = pd.DataFrame([user_input])
        processed = preprocess_input(df_input)
        prob = model.predict_proba(processed)[0, 1]
        label = "Likely Heart Disease" if prob >= 0.5 else "Unlikely Heart Disease"

        st.success(f"Prediction: {label}")
        st.write(f"Probability: {prob:.2f}")

else:
    st.subheader("Upload CSV for Batch Predictions")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Uploaded file preview:", df.head())

        processed = preprocess_input(df)
        probs = model.predict_proba(processed)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df["Prediction"] = preds
        df["Probability"] = probs

        st.write("Predictions:", df.head())

        # Visualization: Bar chart of predictions
        st.bar_chart(df["Prediction"].value_counts())

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Predictions CSV",
            csv,
            "predictions.csv",
            "text/csv"
        )
