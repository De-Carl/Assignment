import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from scipy.sparse import hstack, csr_matrix

# Set page configuration
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🎓",
    layout="wide"
)

class CareerAdvisorWeb:
    def __init__(self, artifacts_dir='artifacts'):
        self.artifacts_dir = artifacts_dir
        self.artifacts = {}
        
    def load_artifacts(self):
        """Load model and encoders"""
        required_artifacts = [
            'model', 'mlb_encoder', 'ohe_encoder', 
            'scaler', 'target_encoder', 'salary_map'
        ]
        try:
            for name in required_artifacts:
                path = os.path.join(self.artifacts_dir, f'{name}.joblib')
                self.artifacts[name] = joblib.load(path)
            return True
        except FileNotFoundError:
            st.error("Error: Model files not found. Please run model_trainer.py first to generate the artifacts folder.")
            return False

    def get_options(self):
        """
        从保存的编码器中提取所有合法的选项。
        这确保了用户只能选择模型“见过”的数据。
        """
        options = {}
        
        # 1. 获取所有技能 (从 MultiLabelBinarizer)
        options['skills'] = list(self.artifacts['mlb_encoder'].classes_)
        
        # 2. 获取分类特征的选项 (从 OneHotEncoder)
        # 我们需要解析 feature_names_out 来还原原始类别
        # 这是一个技巧：OHE 的特征名通常是 "column_value"
        ohe_features = self.artifacts['ohe_encoder'].get_feature_names_out()
        
        country_feats = [f for f in ohe_features if 'country_' in f]
        # 去掉前缀 "country_" 得到真实国家名
        options['locations'] = [f.replace('country_', '') for f in country_feats]

        # 提取 Experience Level
        exp_feats = [f for f in ohe_features if 'experience_level_standardized_' in f]
        options['experience'] = [f.replace('experience_level_standardized_', '') for f in exp_feats]
        
        # 提取 Employment Type
        emp_feats = [f for f in ohe_features if 'employment_type_standardized_' in f]
        options['employment'] = [f.replace('employment_type_standardized_', '') for f in emp_feats]
        
        return options

    def process_input(self, user_input):
        """Process user input"""
        # Simulate current time features
        now = datetime.now()
        # Assume baseline time is 2023-01-01
        time_index = (now - datetime(2023, 1, 1)).days 
        
        # 1. Numerical features DataFrame
        numerical_data = pd.DataFrame([{
            'salary_avg_usd': user_input['expected_salary'],
            'post_year': now.year,
            'post_month': now.month,
            'day_of_week': now.weekday(),
            'time_index': time_index
        }])
        
        # 2. Categorical features DataFrame
        categorical_data = pd.DataFrame([{
            'employment_type_standardized': user_input['employment_type'],
            'experience_level_standardized': user_input['experience_level'],
            'country': user_input['location'] 
        }])
        
        # 3. Encoding
        skills_encoded = self.artifacts['mlb_encoder'].transform([user_input['skills']])
        categoricals_encoded = self.artifacts['ohe_encoder'].transform(categorical_data)
        numerical_features_scaled = self.artifacts['scaler'].transform(numerical_data)
        
        # 4. Combine
        X_encoded = hstack([
            skills_encoded, 
            categoricals_encoded, 
            csr_matrix(numerical_features_scaled)
        ])
        
        return X_encoded

    def predict(self, user_input):
        """Predict and sort results"""
        X = self.process_input(user_input)
        model = self.artifacts['model']
        target_encoder = self.artifacts['target_encoder']
        salary_map = self.artifacts['salary_map']
        
        # Get probabilities
        probs = model.predict_proba(X)[0]
        # Get top 5
        top_5_indices = probs.argsort()[-5:][::-1]
        
        results = []
        for idx in top_5_indices:
            job_name = target_encoder.inverse_transform([idx])[0]
            prob = probs[idx]
            
            # Look up salary
            real_salary = salary_map.get((job_name, user_input['location']), None)
            
            results.append({
                'Recommended Position': job_name,
                'AI Match Score': prob,
                'Local Median Salary (USD)': real_salary
            })
            
        # Sort by salary (None goes last)
        results.sort(key=lambda x: x['Local Median Salary (USD)'] if x['Local Median Salary (USD)'] else -1, reverse=True)
        return results

# --- Streamlit Main Program Logic ---

# 1. Initialize system
@st.cache_resource
def get_advisor():
    advisor = CareerAdvisorWeb()
    if advisor.load_artifacts():
        return advisor
    return None

advisor = get_advisor()

if advisor:
    # Get valid options list
    options = advisor.get_options()

    # --- Sidebar: User Input ---
    with st.sidebar:
        st.header("🧑‍💼 Candidate Profile Settings")
        
        # Skill selection (multi-select, completely solves invalid input)
        selected_skills = st.multiselect(
            "1. Skills Mastered (Multiple Selection)",
            options=options['skills'],
            default=['Python', 'SQL'] # Default value
        )
        
        selected_location = st.selectbox(
            "2. Target Country", 
            options=options['locations']
        )
        
        # Experience selection
        selected_exp = st.selectbox(
            "3. Experience Level",
            options=options['experience']
        )
        
        # Employment type
        selected_emp = st.selectbox(
            "4. Employment Type",
            options=options['employment']
        )
        
        # Salary expectation (slider)
        selected_salary = st.slider(
            "5. Expected Annual Salary (USD)",
            min_value=30000,
            max_value=300000,
            value=120000,
            step=5000
        )
        
        run_btn = st.button("🚀 Start AI Prediction", type="primary")

    # --- Main Page: Results Display ---
    st.title("🤖 AI Career Development Advisor System")
    st.subheader("Nobody predicts better than me!")
    st.markdown("""
    This system uses machine learning models to analyze your skill set and expectations, combined with **32,000+ real job posting data**,
    to recommend the most matching and most lucrative positions for you.
    """)
    
    st.divider()

    if run_btn:
        if not selected_skills:
            st.warning("⚠️ Please select at least one skill!")
        else:
            # Build input dictionary
            user_input = {
                'skills': selected_skills,
                'location': selected_location,
                'experience_level': selected_exp,
                'employment_type': selected_emp,
                'expected_salary': selected_salary
            }
            
            with st.spinner('Analyzing skill matrix and market data...'):
                # Run prediction
                predictions = advisor.predict(user_input)
                
                # Convert results to DataFrame for display
                df_res = pd.DataFrame(predictions)
                
                # Format data
                df_res['AI Match Score'] = df_res['AI Match Score'].apply(lambda x: f"{x:.1%}")
                df_res['Local Median Salary (USD)'] = df_res['Local Median Salary (USD)'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "Insufficient Data")
                
                # --- Display Results ---
                st.success(f"✅ Analysis complete! Found the following recommendations for you in **{selected_location}**:")
                
                # Highlight best recommendation
                best_job = df_res.iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Top Recommended Position", best_job['Recommended Position'])
                with col2:
                    st.metric("Estimated Annual Salary", best_job['Local Median Salary (USD)'])
                with col3:
                    st.metric("AI Match Index", best_job['AI Match Score'])
                
                # Display full table
                st.table(df_res)
                
                st.info("💡 **Tip**: Rankings are based on both your skill match score and the actual salary level for the position in the local area.")

else:
    st.stop()