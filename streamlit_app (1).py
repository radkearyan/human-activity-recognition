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

# Activity visual indicator helper
def get_activity_emoji(activity):
    icons = {
        'WALKING': '🚶 (Walking)',
        'WALKING_UPSTAIRS': '🚶‍♂️ climb (Walking Upstairs)',
        'WALKING_DOWNSTAIRS': '🚶‍♂️ descend (Walking Downstairs)',
        'SITTING': '🪑 (Sitting)',
        'STANDING': '🧍 (Standing)',
        'LAYING': '🛌 (Laying)'
    }
    return icons.get(activity, "❓ (Unknown)")

# Create two sections: Test Dataset sample loading vs Manual adjustment
st.subheader("1. Provide Sensor Measurements")

option = st.radio("How would you like to input the 561 features?", ["Load Real Activity Sample from Test Set", "Input Manually (Sample Values)"])

# Initialize state keys
if "features" not in st.session_state:
    st.session_state.features = np.zeros(561)
if "true_activity" not in st.session_state:
    st.session_state.true_activity = None

if option == "Load Real Activity Sample from Test Set":
    st.info("We will pull a real physical sensor sample from the test dataset to let you verify accuracy!")
    
    # Selected activity to load
    target_act = st.selectbox("Select an activity to simulate:", list(activity_labels.values()))
    
    if st.button("🎲 Load Selected Activity Sample"):
        try:
            # Fetch indices matching chosen class from notebook globals if available
            # Or run a fallback locally
            import pandas as pd
            X_test_df = pd.read_csv('/content/UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
            y_test_df = pd.read_csv('/content/UCI HAR Dataset/test/y_test.txt', delim_whitespace=True, header=None, names=['Activity'])
            
            # Find key matching activity label value
            class_id = [k for k, v in activity_labels.items() if v == target_act][0]
            matching_indices = y_test_df[y_test_df['Activity'] == class_id].index
            
            if len(matching_indices) > 0:
                random_idx = np.random.choice(matching_indices)
                st.session_state.features = X_test_df.iloc[random_idx].values
                st.session_state.true_activity = target_act
                st.success(f"Successfully loaded a real sample of {target_act}!")
            else:
                st.error("No matching samples found.")
        except Exception as e:
            st.error(f"Error loading dataset samples: {e}. Falling back to standard mock arrays.")
            st.session_state.features = np.random.uniform(-0.5, 0.5, 561)
            st.session_state.true_activity = target_act

else:
    st.write("Adjust a subset of primary acceleration features below:")
    f1 = st.slider("Mean Body Acceleration X", -1.0, 1.0, 0.27)
    f2 = st.slider("Mean Body Acceleration Y", -1.0, 1.0, -0.02)
    f3 = st.slider("Mean Body Acceleration Z", -1.0, 1.0, -0.11)
    
    st.session_state.features[0] = f1
    st.session_state.features[1] = f2
    st.session_state.features[2] = f3
    st.session_state.true_activity = None

# Display target validation card
if st.session_state.true_activity:
    st.markdown(f"### **Expected Target Activity:**  
## {get_activity_emoji(st.session_state.true_activity)}")

# 2. Inference Section
st.subheader("2. Activity Classification")

if st.button("🔍 Predict Activity"):
    input_array = np.array(st.session_state.features).reshape(1, -1)
    prediction = int(pipeline.predict(input_array)[0])
    predicted_activity = activity_labels[prediction]
    
    # Display Result
    st.metric(label="Predicted Class", value=predicted_activity)
    
    # Match verification banner
    if st.session_state.true_activity:
        if predicted_activity == st.session_state.true_activity:
            st.success("✅ Verification SUCCESS! The Model correctly identified this activity.")
        else:
            st.warning("⚠️ Verification MISMATCH! Model predicted a different state.")
    else:
        if 'WALKING' in predicted_activity:
             st.info("Dynamic State detected: The user is actively moving.")
        else:
             st.success("Static State detected: The user is resting or stationary.")

st.divider()
with st.expander("View Current 561 Feature Array"):
    st.dataframe(pd.DataFrame(st.session_state.features, columns=["Value"])) 
