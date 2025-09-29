import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import time
import altair as alt # For visualization

# Set to suppress warning when running the app
pd.options.mode.chained_assignment = None

# Define a clean color palette
COLOR_HIGH_STRESS = "#ef4444" # Red
COLOR_LOW_STRESS = "#10b981"  # Green
COLOR_NEUTRAL = "#3b82f6"     # Blue

# --- 1. Data Loading, Preprocessing, and Model Training ---

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads, cleans, encodes, and prepares the Stress dataset."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Stress.csv not found. Please ensure the file is correctly uploaded.")
        return None, None, None, None

    # 1. Clean Column Names
    new_columns = [
        'Age', 'Gender', 'University', 'Department', 'Academic_Year', 'CGPA', 'Waiver',
        'Q1_Upset_Academics', 'Q2_Control_Academics', 'Q3_Nervous_Stressed', 'Q4_Cope_Activities',
        'Q5_Confident_Ability', 'Q6_Life_Going_Way', 'Q7_Control_Irritations', 'Q8_Academic_Performance_Top',
        'Q9_Angered_Low_Grades', 'Q10_Difficulties_Piling_Up', 'Stress_Value', 'Stress_Label'
    ]
    df.columns = new_columns
    
    # 2. Target Variable (Binary Classification)
    # Target = 1 (High Stress) if 'High Perceived Stress', 0 otherwise.
    df['Is_High_Stress'] = (df['Stress_Label'] == 'High Perceived Stress').astype(int)

    # 3. Feature Selection & Transformation
    # Identify positive sentiment questions (high score means LOW stress)
    positive_qs = ['Q5_Confident_Ability', 'Q6_Life_Going_Way', 'Q7_Control_Irritations', 'Q8_Academic_Performance_Top']
    # Invert scores (scale 1-4) so high score always means HIGH stress: New = 5 - Old
    for col in positive_qs:
        df[col] = 5 - df[col]
    
    # Encode Categorical Features for the Model
    # Gender (Label Encoding: Male=0, Female=1)
    df['Gender_Encoded'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    
    # CGPA (Ordinal Encoding)
    cgpa_mapping = {'Below 2.50': 0, '2.50 - 2.99': 1, '3.00 - 3.39': 2, '3.40 - 3.79': 3, '3.80 - 4.00': 4}
    df['CGPA_Encoded'] = df['CGPA'].map(cgpa_mapping)

    # Final feature list (using all 10 inverted questions + encoded demographics)
    feature_cols = [
        'Gender_Encoded', 'CGPA_Encoded',
        'Q1_Upset_Academics', 'Q2_Control_Academics', 'Q3_Nervous_Stressed', 'Q4_Cope_Activities',
        'Q5_Confident_Ability', 'Q6_Life_Going_Way', 'Q7_Control_Irritations', 'Q8_Academic_Performance_Top',
        'Q9_Angered_Low_Grades', 'Q10_Difficulties_Piling_Up'
    ]
    
    # Clean up the dataset to only include relevant columns for training/prediction
    model_df = df[feature_cols + ['Is_High_Stress']].dropna()
    
    return df, model_df, feature_cols, cgpa_mapping

@st.cache_resource
def train_model(model_df, feature_cols):
    """Trains a Logistic Regression model on the processed data."""
    X = model_df[feature_cols]
    y = model_df['Is_High_Stress']

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train a Logistic Regression model (good for interpretability)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    return model

# --- Load Data and Train Model ---
FILE_PATH = "Stress.csv"
full_df, model_df, feature_cols, cgpa_mapping = load_and_preprocess_data(FILE_PATH)

if model_df is not None:
    model = train_model(model_df, feature_cols)
    
    # Extract Q-features for easy iteration in UI
    q_features = [col for col in feature_cols if col.startswith('Q')]
else:
    # Use placeholders if data loading failed
    model, q_features = None, []

# --- 2. Helper Functions for Stress Solutions ---

def get_stress_solution(stress_level, user_inputs, q_features):
    """Provides tailored advice based on predicted stress level and inputs, focusing on high-scoring questions."""
    solutions = []

    if stress_level == 1:
        solutions.append("🛑 **High Perceived Stress Detected!** Your digital twin indicates academic strain is likely high.")

        # Identify the top 3 highest scoring stress factors (questions with scores of 4 or 3)
        stress_factors = {
            'Q1_Upset_Academics': "frequently feeling upset due to academic affairs.",
            'Q2_Control_Academics': "feeling unable to control important things in your academic life.",
            'Q3_Nervous_Stressed': "feeling nervous and stressed due to academic pressure.",
            'Q4_Cope_Activities': "feeling unable to cope with all mandatory academic activities (assignments, exams, etc.).",
            'Q9_Angered_Low_Grades': "getting angered due to bad performance or low grades.",
            'Q10_Difficulties_Piling_Up': "feeling like academic difficulties are piling up.",
            # Note: For inverted positive questions, a score of 4 means the original score was 1 (least confident/most stressed)
            'Q5_Confident_Ability': "low confidence in your ability to handle academic problems.",
            'Q6_Life_Going_Way': "feeling things in your academic life are NOT going your way.",
            'Q7_Control_Irritations': "low ability to control irritations in academic affairs.",
            'Q8_Academic_Performance_Top': "feeling your academic performance is NOT on top."
        }
        
        # Get user's question inputs and sort by score (descending)
        q_scores = {k: user_inputs[k] for k in q_features}
        sorted_q_scores = sorted(q_scores.items(), key=lambda item: item[1], reverse=True)
        
        # The item[0] is the question name (string), item[1] is the score (int)
        top_factors = [q for q, score in sorted_q_scores if score >= 3] # Focus on scores of 3 or 4

        if top_factors:
            solutions.append("🎯 **Primary Areas of Concern:** Based on your inputs, we should address:")
            
            # Use the full question name string (q_name) as the key
            for q_name in top_factors[:3]: 
                solutions.append(f"- **{stress_factors[q_name].capitalize()}**") 

            # Provide tailored advice based on the top factor category
            if 'Q4_Cope_Activities' in top_factors or 'Q10_Difficulties_Piling_Up' in top_factors:
                solutions.append("⏳ **Time Management Solution:** Use a structured weekly planner. Break large assignments (Q4, Q10) into small, 30-minute tasks. Celebrate small victories!")
            
            elif 'Q3_Nervous_Stressed' in top_factors or 'Q9_Angered_Low_Grades' in top_factors:
                solutions.append("🧘 **Emotional Regulation Solution:** When stress (Q3) or anger (Q9) spikes, try the 4-7-8 breathing technique (inhale 4, hold 7, exhale 8). This quickly engages your parasympathetic nervous system.")

            elif 'Q2_Control_Academics' in top_factors or 'Q5_Confident_Ability' in top_factors:
                solutions.append("💡 **Focus on Control Solution:** You feel a lack of control (Q2, Q5). Focus only on what you *can* control: your effort, your study method, and asking for help. Delegate or defer what you cannot.")

        else:
            # Fallback if prediction is high but individual scores were moderate
            solutions.append("🧐 **Holistic Check:** Though no single factor stands out, the overall pattern suggests stress. Ensure you're eating well, sleeping 7+ hours, and taking non-academic social breaks.")


    else:
        solutions.append("✅ **Low/Moderate Stress Indication.** Your academic metrics suggest you are managing pressure effectively. Keep up the great work!")
        solutions.append("📈 **Optimize & Maintain:** While stress is low, monitor your inputs daily. Consider implementing a 'future-self' habit (like preparing for the next day's classes) to maintain this low stress level.")

    return solutions

# --- 3. Streamlit Application Layout ---

# Configuration and Styling
st.set_page_config(
    page_title="Digital Twin: Academic Stress Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    .reportview-container .main {{
        padding-top: 2rem;
    }}
    .stApp {{
        background-color: #f0f2f6;
    }}
    .header-text {{
        color: {COLOR_NEUTRAL};
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }}
    .stSidebar [data-testid="stSidebarContent"] {{
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
    }}
    .stress-high {{
        background-color: #fef2f2; /* Light Red */
        color: {COLOR_HIGH_STRESS}; /* Red Text */
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #f87171;
    }}
    .stress-low {{
        background-color: #ecfdf5; /* Light Green */
        color: {COLOR_LOW_STRESS}; /* Green Text */
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #34d399;
    }}
    h1 {{
        font-family: 'Inter', sans-serif;
        color: #1e3a8a;
    }}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Digital Twin: Academic Stress Analyzer")
st.markdown("<p class='header-text'>Based on your survey responses, we predict your current level of perceived academic stress and provide targeted solutions.</p>", unsafe_allow_html=True)

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("📝 Stress Scale Input (1 = Never, 4 = Very Often)")
    st.markdown("Please score how often you've experienced the following in a semester:")
    
    # Demographic Input
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ('Female', 'Male'))
    cgpa_input = st.selectbox("Current CGPA Range", list(cgpa_mapping.keys()))
    
    # Added clear subheader and helper text for clarity
    st.subheader("Academic Stressors (High Score = High Stress)")
    st.markdown("*(Q1, Q2, Q3, Q4, Q9, Q10: Score 4 means **Very Often Stressed**)*")
    
    # Question Inputs (Negative Sentiment - High score = High stress)
    q1 = st.slider("Q1: Upset due to academic affairs?", 1, 4, 2)
    q2 = st.slider("Q2: Unable to control important things?", 1, 4, 2)
    q3 = st.slider("Q3: Nervous and stressed by pressure?", 1, 4, 2)
    q4 = st.slider("Q4: Unable to cope with activities?", 1, 4, 2)
    q9 = st.slider("Q9: Angered due to low grades?", 1, 4, 2)
    q10 = st.slider("Q10: Difficulties piling up?", 1, 4, 2)
    
    # Question Inputs (Positive Sentiment - High score = Low stress)
    # The UI shows the original question, but the transformation is applied below
    # Added clear subheader and helper text for clarity
    st.subheader("Confidence/Control (High Score = Low Stress)")
    st.markdown("*(Q5, Q6, Q7, Q8: Score 4 means **Very Often Confident**)*")
    
    # IMPORTANT: Default to 1 here means minimum confidence/control, which inverts to max stress for these factors.
    q5 = st.slider("Q5: Confident about your ability to handle problems?", 1, 4, 1) 
    q6 = st.slider("Q6: Things in academic life going your way?", 1, 4, 1) 
    q7 = st.slider("Q7: Able to control irritations?", 1, 4, 1) 
    q8 = st.slider("Q8: Academic performance was on top?", 1, 4, 1)

    # Transform inputs for the model based on preprocessing logic
    q5_inv = 5 - q5
    q6_inv = 5 - q6
    q7_inv = 5 - q7
    q8_inv = 5 - q8
    
    cgpa_enc = cgpa_mapping.get(cgpa_input, 0)
    gender_enc = 1 if gender == 'Female' else 0

    # Map inputs to a dictionary corresponding to the feature_cols list
    input_data = {
        'Gender_Encoded': gender_enc,
        'CGPA_Encoded': cgpa_enc,
        'Q1_Upset_Academics': q1,
        'Q2_Control_Academics': q2,
        'Q3_Nervous_Stressed': q3,
        'Q4_Cope_Activities': q4,
        'Q5_Confident_Ability': q5_inv, # Inverted score for model
        'Q6_Life_Going_Way': q6_inv,     # Inverted score for model
        'Q7_Control_Irritations': q7_inv, # Inverted score for model
        'Q8_Academic_Performance_Top': q8_inv, # Inverted score for model
        'Q9_Angered_Low_Grades': q9,
        'Q10_Difficulties_Piling_Up': q10
    }
    
    if st.button("Analyze Stress State", use_container_width=True, type="primary"):
        st.session_state.run_analysis = True
    
    st.markdown("""
        ---
        <small>Model used: Logistic Regression (trained on **Stress.csv** data).
        A high score on Q5-Q8 indicates low stress; scores are automatically inverted for prediction.</small>
    """, unsafe_allow_html=True)

# Initialize session state for button click
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# --- 4. Main Display Area for Digital Twin Result ---
st.header("🔬 Stress Prediction Digital Twin")

if model is None:
    st.error("Model could not be initialized. Please check the file path and data content.")
elif st.session_state.run_analysis:
    # 4.1 Prediction Logic
    
    # Use a spinner to simulate a digital twin 'scan' or analysis
    with st.spinner('Analyzing metrics and simulating academic stress state...'):
        time.sleep(1.5) # Simulate processing time
        
        # FIX FOR ERROR: Explicitly create the prediction DataFrame using the feature_cols list 
        # to guarantee the correct number and order of columns for the model.
        input_for_prediction = pd.DataFrame([input_data], columns=feature_cols)
        
        prediction_proba = model.predict_proba(input_for_prediction)[0]
        prediction = model.predict(input_for_prediction)[0]

    stress_status = "High Stress" if prediction == 1 else "Low/Moderate Stress"
    stress_class = "stress-high" if prediction == 1 else "stress-low"
    
    # Use two columns for prediction and solution
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Predicted Stress Level")
        st.markdown(f"""
            <div class='{stress_class} text-center'>
                <h1 style='color:inherit; font-size: 3rem; margin-top:0;'>{stress_status}</h1>
                <p style='margin-bottom:0;'>Confidence: {prediction_proba[prediction]:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # --- Visualization 1: Prediction Confidence (Donut Chart) ---
        chart_data = pd.DataFrame({'status': [stress_status, "Other"], 'value': [prediction_proba[prediction], 1 - prediction_proba[prediction]]})
        
        base = alt.Chart(chart_data).encode(
            theta=alt.Theta("value", stack=True)
        )

        pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
            color=alt.Color("status", scale=alt.Scale(domain=["High Stress", "Low/Moderate Stress", "Other"], range=[COLOR_HIGH_STRESS, COLOR_LOW_STRESS, "#e5e7eb"])),
            order=alt.Order("value", sort="descending"),
            tooltip=["status", alt.Tooltip("value", format=".1%")]
        ).properties(title="Model Confidence")
        
        st.altair_chart(pie, use_container_width=True)


    with col2:
        st.header("💡 Personalized Stress Solutions")
        solutions = get_stress_solution(prediction, input_data, q_features)
        
        for sol in solutions:
            st.markdown(f"- {sol}")

    st.markdown("---")
    
    # --- Visualization 2: User Input Score Profile (Bar Chart) ---
    st.subheader("Your Input Profile: Where is Your Stress Highest?")
    
    # Define cleaner display names for the Q-features in the input chart
    question_display_map = {
        'Q1_Upset_Academics': '1. Upset (Acad)',
        'Q2_Control_Academics': '2. Lack Control (Acad)',
        'Q3_Nervous_Stressed': '3. Nervous/Stressed',
        'Q4_Cope_Activities': '4. Can\'t Cope (Act)',
        'Q5_Confident_Ability': '5. Low Confidence', 
        'Q6_Life_Going_Way': '6. Life NOT Going Way', 
        'Q7_Control_Irritations': '7. Low Control Irritations', 
        'Q8_Academic_Performance_Top': '8. Performance NOT Top', 
        'Q9_Angered_Low_Grades': '9. Angered by Low Grades',
        'Q10_Difficulties_Piling_Up': '10. Difficulties Piling Up',
    }

    # Prepare data for user input chart (using INVERTED Q5-Q8)
    input_scores_df = pd.DataFrame({
        'Question': [question_display_map.get(q, q) for q in q_features],
        'Score': [input_data[q] for q in q_features]
    })
    
    input_chart = alt.Chart(input_scores_df).mark_bar().encode(
        x=alt.X('Score', title='User Score (1=Low Stress, 4=High Stress)', scale=alt.Scale(domain=[1, 4])),
        y=alt.Y('Question', sort=alt.EncodingSortField(field="Score", op="max", order='descending'), title=''), # Ensure no y-axis title to save space
        color=alt.condition(
            alt.datum.Score >= 3,
            alt.value(COLOR_HIGH_STRESS), # Highlight high stress factors
            alt.value(COLOR_NEUTRAL)   # Default color
        ),
        tooltip=['Question', 'Score']
    ).properties(height=350) # Increased height to ensure all labels fit
    st.altair_chart(input_chart, use_container_width=True)

    # Reset state for next analysis
    st.session_state.run_analysis = False
    
# --- 5. Model Insights Section (Feature Importance & Global Context) ---
st.markdown("---")
st.header("📊 Model Insights and Data Context")

if model is not None:
    # --- Visualization 3: Global Stress Distribution by CGPA (Dataset Context) ---
    if full_df is not None:
        st.subheader("Global Data Context: High Stress Distribution by CGPA Range")
        
        # Calculate stress distribution per CGPA group
        cgpa_stress_dist = model_df.groupby('CGPA_Encoded')['Is_High_Stress'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
        # Remap encoded values back to readable ranges
        reverse_cgpa_mapping = {v: k for k, v in cgpa_mapping.items()}
        cgpa_stress_dist['CGPA_Range'] = cgpa_stress_dist['CGPA_Encoded'].map(reverse_cgpa_mapping)
        cgpa_stress_dist['Stress_Level'] = cgpa_stress_dist['Is_High_Stress'].map({1: 'High Stress', 0: 'Low/Moderate Stress'})

        # Altair Stacked Bar Chart
        global_chart = alt.Chart(cgpa_stress_dist).mark_bar().encode(
            x=alt.X('CGPA_Range', sort=list(cgpa_mapping.keys()), title='CGPA Range (From Dataset)'),
            y=alt.Y('Percentage', stack="normalize", axis=alt.Axis(format='%', title='Percentage of Students')),
            color=alt.Color('Stress_Level', scale=alt.Scale(domain=['High Stress', 'Low/Moderate Stress'], range=[COLOR_HIGH_STRESS, COLOR_LOW_STRESS]), legend=alt.Legend(title="Stress Level")),
            order=alt.Order('Stress_Level', sort='descending'),
            tooltip=['CGPA_Range', 'Stress_Level', alt.Tooltip('Percentage', format='.1f')]
        ).properties(height=300)
        
        st.altair_chart(global_chart, use_container_width=True)
        st.caption("This chart shows the stress levels across different CGPA groups in the original data used for training.")
        st.markdown("---")

    # --- Visualization 4: Feature Importance (Bar Chart) ---
    try:
        # Create a series of coefficients (feature importance for Logistic Regression)
        coefficients = pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=False)
        
        # Rename features for better display in the Feature Importance chart
        display_names = {
            'Gender_Encoded': 'Gender (Female: Higher Risk)',
            'CGPA_Encoded': 'CGPA (High: Lower Risk)', 
            'Q1_Upset_Academics': 'Q1: Upset Academics',
            'Q2_Control_Academics': 'Q2: Lack of Control',
            'Q3_Nervous_Stressed': 'Q3: Nervous/Stressed',
            'Q4_Cope_Activities': 'Q4: Can\'t Cope',
            'Q5_Confident_Ability': 'Q5: Low Confidence (Inv)',
            'Q6_Life_Going_Way': 'Q6: Life NOT Going Way (Inv)',
            'Q7_Control_Irritations': 'Q7: Low Control Irritations (Inv)',
            'Q8_Academic_Performance_Top': 'Q8: Performance NOT Top (Inv)',
            'Q9_Angered_Low_Grades': 'Q9: Angered by Low Grades',
            'Q10_Difficulties_Piling_Up': 'Q10: Difficulties Piling Up',
        }
        
        coefficients.index = coefficients.index.map(lambda x: display_names.get(x, x))
        
        st.subheader("Model Feature Impact on High Stress Likelihood")
        
        # Plot coefficients using Altair for better styling
        chart_data = coefficients.reset_index()
        chart_data.columns = ['Feature', 'Impact']
        
        # Calculate max absolute impact for symmetric scaling to ensure the chart is visually centered
        max_abs_impact = chart_data['Impact'].abs().max()
        x_domain = [-max_abs_impact * 1.05, max_abs_impact * 1.05] # Add a small buffer for visualization
        
        chart = alt.Chart(chart_data).mark_bar().encode(
            # Using symmetric domain to ensure the zero-line is centered visually
            x=alt.X('Impact', title='Coefficient (Impact on High Stress)', scale=alt.Scale(domain=x_domain)),
            y=alt.Y('Feature', sort='-x', title=''), # Ensure no y-axis title to save space
            color=alt.condition(
                alt.datum.Impact > 0,
                alt.value(COLOR_HIGH_STRESS),  # Red for positive impact (higher stress)
                alt.value(COLOR_LOW_STRESS)   # Green for negative impact (lower stress)
            ),
            tooltip=['Feature', alt.Tooltip('Impact', format='.3f')]
        ).properties(height=400) # Increased height significantly
        
        st.altair_chart(chart, use_container_width=True)
        
        st.markdown("""
        * **Positive Impact (Red)**: An increase in this factor strongly **increases** the likelihood of **High Stress**.
        * **Negative Impact (Green)**: An increase in this factor **decreases** the likelihood of **High Stress** (e.g., higher CGPA).
        """)
        
    except Exception as e:
        st.warning(f"Could not display model coefficients. This usually happens if the model failed to train or if data columns are mismatched. Error: {e}")
