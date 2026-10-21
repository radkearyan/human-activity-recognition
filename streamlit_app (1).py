import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Human Activity Recognition (HAR)",
    page_icon="🏃",
    layout="centered"
)

st.title("🏃 Human Activity Recognition Predictor")
st.write("This interactive app uses the tuned **Linear SVM** model to predict user physical activity from smartphone sensor readings.")

# 1. Load the pipeline
@st.cache_resource
def load_model():
    return joblib.load('har_svm_pipeline.joblib')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading the model artifact: {e}")
    st.stop()

activity_labels = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

# Create two sections: Manual adjustment vs Mock generation
st.subheader("1. Provide Sensor Measurements")

option = st.radio("How would you like to input the 561 features?", ["Generate Random Activity Features", "Input Manually (Sample Values)"])

# Initialize state for features
if "features" not in st.session_state:
    # Initialize with zeros
    st.session_state.features = np.zeros(561)

if option == "Generate Random Activity Features":
    if st.button("🎲 Generate Mock Activity Data"):
        # Generate realistic looking values within [-1.0, 1.0] as standardized in HAR
        st.session_state.features = np.random.uniform(-0.8, 0.8, 561)
        st.success("Mock features generated successfully!")
else:
    st.write("Adjust a subset of primary acceleration features below:")
    # Display sliders for the first 3 variables (mean body acceleration X, Y, Z)
    f1 = st.slider("Mean Body Acceleration X", -1.0, 1.0, 0.27)
    f2 = st.slider("Mean Body Acceleration Y", -1.0, 1.0, -0.02)
    f3 = st.slider("Mean Body Acceleration Z", -1.0, 1.0, -0.11)
    
    st.session_state.features[0] = f1
    st.session_state.features[1] = f2
    st.session_state.features[2] = f3

# 2. Inference Section
st.subheader("2. Activity Classification")

if st.button("🔍 Predict Activity"):
    input_array = np.array(st.session_state.features).reshape(1, -1)
    prediction = int(pipeline.predict(input_array)[0])
    predicted_activity = activity_labels[prediction]
    
    # Display Result
    st.metric(label="Predicted Class", value=predicted_activity)
    
    # Display contextual indicators based on activity
    if 'WALKING' in predicted_activity:
         st.info("Dynamic State detected: The user is actively moving.")
    else:
         st.success("Static State detected: The user is resting or stationary.")

st.divider()
with st.expander("View Current 561 Feature Array"):
    st.dataframe(pd.DataFrame(st.session_state.features, columns=["Value"])) 
