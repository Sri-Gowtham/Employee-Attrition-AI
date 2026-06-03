"""
app.py
------
Streamlit application for Employee Attrition Prediction.
"""

import sys
import json
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from predict import load_pipeline, predict_single, predict_batch
from data_preprocessing import EXPECTED_DEPARTMENTS, EXPECTED_SALARY_LEVELS

# ---------------------------------------------------------------------------
# Config & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Workforce Analytics Platform V4",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #ff4b4b;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        color: white;
    }
    .metric-title {
        font-size: 14px;
        color: #a0aec0;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper: Risk Level
# ---------------------------------------------------------------------------
def get_risk_level(prob):
    if prob <= 0.3:
        return "Low Risk", "green"
    elif prob <= 0.7:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"

# ---------------------------------------------------------------------------
# Load Model & Metrics
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    model_path = Path(__file__).parent.parent / "models" / "attrition_rf_pipeline.joblib"
    try:
        return load_pipeline(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = get_model()

@st.cache_data
def get_metrics():
    metrics_path = Path(__file__).parent.parent / "models" / "evaluation_metrics.json"
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return None

metrics = get_metrics()

@st.cache_data
def get_analytics_data():
    """Loads dataset and appends model predictions for analytics."""
    data_path = Path(__file__).parent.parent / "data" / "employee_data.csv"
    try:
        df = pd.read_csv(data_path)
        if pipeline is not None:
            # Predict using the loaded model
            df_pred = predict_batch(pipeline, df)
            # Add risk level string for analytics
            df_pred['Risk_Category'] = df_pred['probability_quit'].apply(lambda p: get_risk_level(p)[0].split(" ")[0])
            return df_pred
        return df
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        return pd.DataFrame()

analytics_df = get_analytics_data()

# ---------------------------------------------------------------------------
# Sidebar Layout
# ---------------------------------------------------------------------------
st.sidebar.header("🎯 Predict Single Employee")

with st.sidebar.form("prediction_form"):
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
    number_project = st.number_input("Number of Projects", 1, 20, 3)
    average_montly_hours = st.number_input("Average Monthly Hours", 50, 400, 150)
    time_spend_company = st.number_input("Years at Company", 1, 40, 3)
    work_accident = st.selectbox("Work Accident?", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years?", ["No", "Yes"])
    department = st.selectbox("Department", EXPECTED_DEPARTMENTS)
    salary = st.selectbox("Salary Level", EXPECTED_SALARY_LEVELS)
    
    submit_button = st.form_submit_button(label="Predict Attrition")

st.sidebar.divider()
st.sidebar.header("📊 Analytics Filters")
st.sidebar.markdown("*(Applies to Workforce Analytics tab)*")

# Dynamic filters based on loaded data
if not analytics_df.empty:
    filter_dept = st.sidebar.multiselect("Department", options=analytics_df['department'].unique(), default=[])
    filter_salary = st.sidebar.multiselect("Salary Level", options=analytics_df['salary'].unique(), default=[])
    filter_promo = st.sidebar.multiselect("Promotion (Last 5 Years)", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No", default=[])
    filter_accident = st.sidebar.multiselect("Work Accident", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No", default=[])
    
    min_years = int(analytics_df['time_spend_company'].min())
    max_years = int(analytics_df['time_spend_company'].max())
    filter_years = st.sidebar.slider("Years at Company", min_years, max_years, (min_years, max_years))

    # Apply filters to analytics dataframe
    filtered_df = analytics_df.copy()
    if filter_dept:
        filtered_df = filtered_df[filtered_df['department'].isin(filter_dept)]
    if filter_salary:
        filtered_df = filtered_df[filtered_df['salary'].isin(filter_salary)]
    if filter_promo:
        filtered_df = filtered_df[filtered_df['promotion_last_5years'].isin(filter_promo)]
    if filter_accident:
        filtered_df = filtered_df[filtered_df['Work_accident'].isin(filter_accident)]
    filtered_df = filtered_df[(filtered_df['time_spend_company'] >= filter_years[0]) & (filtered_df['time_spend_company'] <= filter_years[1])]
else:
    filtered_df = pd.DataFrame()


# Store single prediction results in session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
    st.session_state.employee_data = None

if submit_button:
    if pipeline is not None:
        employee_data = {
            "satisfaction_level": satisfaction_level,
            "last_evaluation": last_evaluation,
            "number_project": number_project,
            "average_montly_hours": average_montly_hours,
            "time_spend_company": time_spend_company,
            "Work_accident": 1 if work_accident == "Yes" else 0,
            "promotion_last_5years": 1 if promotion_last_5years == "Yes" else 0,
            "department": department,
            "salary": salary,
        }
        st.session_state.employee_data = employee_data
        st.session_state.prediction_result = predict_single(pipeline, employee_data)


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.title("💼 Workforce Analytics Platform V4")

tab_home, tab_analytics, tab_single, tab_explain, tab_batch, tab_insights = st.tabs([
    "🏠 Home", 
    "📊 Workforce Analytics",
    "👤 Single Prediction", 
    "🧠 Explain Prediction", 
    "📂 Batch Prediction", 
    "📈 Model Metrics"
])

# --- HOME TAB ---
with tab_home:
    st.markdown("""
    ## Executive Dashboard Overview
    Welcome to the **Workforce Analytics Platform V4**. This application provides comprehensive insights into employee retention, 
    powered by an advanced Random Forest Machine Learning model. 
    
    ### Key Capabilities:
    * **Workforce Analytics:** Executive-level dashboards filtering historical and predicted data.
    * **Single Prediction:** Real-time attrition probability for prospective or current employees.
    * **Explainable AI (SHAP):** Transparent breakdown of exactly why the AI made a specific prediction.
    * **Batch Processing:** Evaluate hundreds of employees instantly via CSV upload.
    """)
    
    st.markdown("### Core Model Performance")
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
        col2.metric("Test F1 Score", f"{metrics.get('test_f1_weighted', 0):.2%}")
        col3.metric("Test ROC AUC", f"{metrics.get('test_roc_auc', 0):.2%}")
    else:
        st.info("Metrics not found.")

# --- WORKFORCE ANALYTICS TAB ---
with tab_analytics:
    if filtered_df.empty:
        st.warning("No data available or filters are too restrictive.")
    else:
        st.subheader("Executive KPIs")
        
        # Calculate KPIs
        total_emp = len(filtered_df)
        predicted_leave = filtered_df['predicted_quit'].sum()
        attrition_rate = predicted_leave / total_emp if total_emp > 0 else 0
        high_risk = len(filtered_df[filtered_df['Risk_Category'] == 'High'])
        avg_sat = filtered_df['satisfaction_level'].mean()
        avg_tenure = filtered_df['time_spend_company'].mean()
        
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Employees", f"{total_emp:,}")
        k2.metric("Predicted to Leave", f"{predicted_leave:,}")
        k3.metric("Attrition Rate", f"{attrition_rate:.1%}")
        k4.metric("High Risk", f"{high_risk:,}")
        k5.metric("Avg Satisfaction", f"{avg_sat:.2f}")
        k6.metric("Avg Tenure (Yrs)", f"{avg_tenure:.1f}")
        
        st.divider()
        
        # Row 1: Dept and Salary
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### A. Department Analytics")
            dept_stats = filtered_df.groupby('department').agg(
                Count=('department', 'count'),
                Attrition=('predicted_quit', 'mean')
            ).reset_index().sort_values('Attrition', ascending=False)
            
            fig_dept = px.bar(dept_stats, x='department', y='Attrition', 
                              hover_data=['Count'], text_auto='.1%',
                              labels={'Attrition': 'Attrition Rate', 'department': 'Department'},
                              title="Attrition Rate by Department")
            st.plotly_chart(fig_dept, use_container_width=True)
            
        with c2:
            st.markdown("#### B. Salary Analytics")
            sal_stats = filtered_df.groupby('salary').agg(
                Count=('salary', 'count'),
                Attrition=('predicted_quit', 'mean')
            ).reset_index()
            
            fig_sal = px.bar(sal_stats, x='salary', y='Attrition', 
                             color='salary', text_auto='.1%',
                             title="Attrition Rate by Salary Level")
            st.plotly_chart(fig_sal, use_container_width=True)
            
        # Row 2: Promo and Satisfaction
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### C. Promotion Analytics")
            promo_stats = filtered_df.groupby(['promotion_last_5years', 'predicted_quit']).size().reset_index(name='Count')
            promo_stats['promotion_last_5years'] = promo_stats['promotion_last_5years'].map({0: 'No', 1: 'Yes'})
            promo_stats['predicted_quit'] = promo_stats['predicted_quit'].map({0: 'Stay', 1: 'Leave'})
            
            fig_promo = px.bar(promo_stats, x='promotion_last_5years', y='Count', color='predicted_quit', 
                               barmode='group', title="Attrition vs Promotion Status (Last 5 Yrs)")
            st.plotly_chart(fig_promo, use_container_width=True)
            
        with c4:
            st.markdown("#### D. Satisfaction Analytics")
            fig_sat = px.box(filtered_df, x='risk_label', y='satisfaction_level', color='risk_label',
                             title="Satisfaction Distribution by Risk Segment")
            st.plotly_chart(fig_sat, use_container_width=True)

        # Row 3: Tenure and Risk
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### E. Tenure Analytics")
            tenure_stats = filtered_df.groupby('time_spend_company')['predicted_quit'].mean().reset_index()
            fig_tenure = px.line(tenure_stats, x='time_spend_company', y='predicted_quit', markers=True,
                                 title="Attrition Rate by Years at Company")
            fig_tenure.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig_tenure, use_container_width=True)
            
        with c6:
            st.markdown("#### F. Risk Analytics")
            risk_counts = filtered_df['Risk_Category'].value_counts().reset_index()
            risk_counts.columns = ['Risk', 'Count']
            color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            fig_risk = px.pie(risk_counts, values='Count', names='Risk', hole=0.4, 
                              color='Risk', color_discrete_map=color_map,
                              title="Employee Risk Segmentation")
            st.plotly_chart(fig_risk, use_container_width=True)

        # AI Insights Panel
        st.divider()
        st.markdown("### 🤖 Automated Executive Insights")
        
        insights = []
        if not dept_stats.empty:
            highest_dept = dept_stats.iloc[0]
            insights.append(f"**Department Risk:** The **{highest_dept['department'].title()}** department shows the highest attrition risk at **{highest_dept['Attrition']:.1%}**.")
        
        if not sal_stats.empty:
            highest_sal = sal_stats.sort_values('Attrition', ascending=False).iloc[0]
            insights.append(f"**Compensation Impact:** Employees in the **{highest_sal['salary'].title()}** salary bracket represent the highest vulnerability group (**{highest_sal['Attrition']:.1%}** attrition).")
            
        low_sat_high_risk = len(filtered_df[(filtered_df['satisfaction_level'] < 0.4) & (filtered_df['predicted_quit'] == 1)])
        if low_sat_high_risk > 0:
            insights.append(f"**Critical Segment:** **{low_sat_high_risk}** employees have low satisfaction (<0.4) and are predicted to leave. This group requires **immediate intervention**.")
            
        insights.append(f"**Overall Health:** The average workforce satisfaction level is **{avg_sat:.2f}/1.00**.")
        
        for ins in insights:
            st.info(ins)


# --- SINGLE PREDICTION TAB ---
with tab_single:
    st.subheader("Single Employee Prediction Results")
    
    result = st.session_state.prediction_result
    if result:
        prob_quit = result['probability_quit']
        risk_label, risk_color = get_risk_level(prob_quit)
        
        st.markdown(f"### Risk Assessment: <span style='color:{risk_color}'>{risk_label}</span>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Prediction",
                value="Leave" if result["prediction"] == 1 else "Stay",
                delta="-At Risk" if result["prediction"] == 1 else "Safe",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Attrition Probability",
                value=f"{prob_quit:.1%}"
            )
        
        with col3:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_quit * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgray"},
                    'steps' : [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "gold"},
                        {'range': [70, 100], 'color': "salmon"}],
                }
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        st.info("💡 Go to the **Explain Prediction** tab to see why the model made this prediction.")
            
    else:
        st.info("👈 Enter employee details in the sidebar and click 'Predict Attrition' to see results.")

# --- EXPLAIN PREDICTION TAB ---
with tab_explain:
    st.subheader("🧠 Explainable AI (SHAP)")
    
    if st.session_state.prediction_result and st.session_state.employee_data:
        try:
            with st.spinner("Generating Explanation..."):
                model = pipeline.named_steps.get('classifier', pipeline.steps[-1][1])
                preprocessor = pipeline.named_steps.get('preprocessor', pipeline.steps[0][1])
                
                df_input = pd.DataFrame([st.session_state.employee_data])
                X_transformed = preprocessor.transform(df_input)
                
                feature_names = preprocessor.get_feature_names_out()
                clean_names = [name.split('__')[-1] for name in feature_names]
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)
                
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]
                    base_value = explainer.expected_value[1]
                else:
                    shap_vals = shap_values[0]
                    base_value = explainer.expected_value
                    if len(shap_vals.shape) > 1 and shap_vals.shape[1] == 2:
                        shap_vals = shap_vals[:, 1]
                        if isinstance(base_value, (list, np.ndarray)):
                            base_value = base_value[1]
                
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = float(base_value[0])
                elif not isinstance(base_value, float):
                    base_value = float(base_value)

                shap_df = pd.DataFrame({
                    'Feature': clean_names,
                    'SHAP Value': shap_vals,
                    'Absolute SHAP': np.abs(shap_vals)
                }).sort_values(by='Absolute SHAP', ascending=False)
                
                top_5 = shap_df.head(5)
                
                prediction_class = "likely to leave" if st.session_state.prediction_result['prediction'] == 1 else "likely to stay"
                
                st.markdown(f"### The employee is {prediction_class} because of:")
                
                for _, row in top_5.iterrows():
                    feat = row['Feature'].replace('_', ' ').title()
                    
                    if st.session_state.prediction_result['prediction'] == 1:
                        if row['SHAP Value'] > 0:
                            st.markdown(f"- **{feat}** pushes towards leaving.")
                        else:
                            st.markdown(f"- **{feat}** helps retain the employee, but isn't enough.")
                    else:
                        if row['SHAP Value'] < 0:
                            st.markdown(f"- **{feat}** pushes towards staying.")
                        else:
                            st.markdown(f"- **{feat}** increases risk slightly, but overall profile is safe.")
                
                st.markdown("---")
                st.markdown("### SHAP Waterfall Plot")
                st.markdown("Shows how each feature moves the model output from the base value to the final prediction.")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                explanation = shap.Explanation(
                    values=shap_vals,
                    base_values=base_value,
                    data=X_transformed[0],
                    feature_names=clean_names
                )
                shap.waterfall_plot(explanation, show=False)
                st.pyplot(fig)
                plt.clf()

                # SHAP Force Plot removed to resolve initjs() compatibility error in Streamlit.
                # Waterfall plot above remains fully functional.

        except Exception as e:
            st.error(f"Error generating explanation: {e}")
            st.exception(e)
    else:
        st.info("Make a prediction in the sidebar to see the explanation here.")

# --- BATCH PREDICTION TAB ---
with tab_batch:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV file containing employee data to predict attrition for multiple employees at once.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            if st.button("Run Batch Prediction"):
                if pipeline is None:
                    st.error("Model not loaded.")
                else:
                    with st.spinner("Predicting..."):
                        results_df = predict_batch(pipeline, df)
                        results_df['risk_level'] = results_df['probability_quit'].apply(lambda p: get_risk_level(p)[0])
                        
                        st.success("Batch prediction complete!")
                        st.dataframe(results_df[["predicted_quit", "probability_quit", "risk_level"]].head(10))
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="attrition_predictions_v4.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- MODEL METRICS TAB ---
with tab_insights:
    st.subheader("Feature Importance")
    if pipeline is not None:
        try:
            model = pipeline.named_steps.get('classifier', pipeline.steps[-1][1])
            preprocessor = pipeline.named_steps.get('preprocessor', pipeline.steps[0][1])
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = preprocessor.get_feature_names_out()
                clean_names = [name.split('__')[-1] for name in feature_names]
                
                imp_df = pd.DataFrame({
                    'Feature': clean_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
                
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                             title="Global Feature Importance in Random Forest Model")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
        except Exception as e:
            st.warning(f"Could not extract feature importance: {e}")
