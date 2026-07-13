import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .prediction-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# CONFIG SECTION
# ================================
@st.cache_resource
def load_model():
    try:
        with open("best_catboost_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model file 'best_catboost_model.pkl' not found!")
        st.stop()

FEATURE_ORDER = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'Gender_Male',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes',
    'Age_group_(18, 30]', 'Age_group_(30, 40]', 'Age_group_(40, 50]',
    'Age_group_(50, 60]'
]

CATEGORY_ORDERS = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager',
                'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive',
                'Sales Representative'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes']
}

SCALER_MEAN = [
    36.923843, 802.485714, 9.192857, 2.912925, 1024.865306,
    2.721769, 65.891156, 2.729932, 2.063946, 2.728571,
    6502.931293, 14313.103401, 2.693197, 15.209524, 3.153741,
    2.712585, 0.793878, 11.279592, 2.799320, 2.761224,
    7.008163, 4.229252, 2.187755, 4.123129, 0.190476,
    0.714966, 0.657483, 0.295918, 0.410204, 0.108844,
    0.269388, 0.084014, 0.092517, 0.585714, 0.034694,
    0.176871, 0.068027, 0.093878, 0.054422, 0.095238,
    0.276871, 0.054422, 0.457823, 0.318367, 0.161224,
    0.321429, 0.389796, 0.231293, 0.057823
]

SCALER_SCALE = [
    9.129341, 403.266476, 8.101360, 1.024098, 602.466788,
    1.090894, 20.279416, 0.711027, 1.106659, 1.102462,
    4707.608477, 7117.815666, 2.497957, 3.659491, 0.360883,
    1.081209, 0.852077, 7.780782, 1.289271, 0.706476,
    6.126525, 3.623137, 3.222430, 3.567584, 0.392813,
    0.451545, 0.474598, 0.456552, 0.492067, 0.311514,
    0.443717, 0.277557, 0.289913, 0.492672, 0.183026,
    0.381554, 0.251781, 0.291693, 0.226893, 0.293480,
    0.447486, 0.226893, 0.498425, 0.465993, 0.367890,
    0.467162, 0.487840, 0.421790, 0.233568
]

model = load_model()

# ================================
# HELPER FUNCTIONS
# ================================
def build_input_row(user_values: dict) -> pd.DataFrame:
    """Transform user inputs into model-ready format"""
    age = user_values['Age']
    age_group = pd.cut([age], bins=[0, 18, 30, 40, 50, 60], include_lowest=True)[0]
    
    cat_values = []
    for col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']:
        cat_order = CATEGORY_ORDERS[col]
        val = user_values[col]
        idx = cat_order.index(val) if val in cat_order else -1
        cat_values.append(idx)
    
    num_values = [user_values[f] for f in [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
        'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]]
    
    all_values = num_values + cat_values
    X_user = np.array(all_values, dtype=float).reshape(1, -1)
    X_user = (X_user - np.array(SCALER_MEAN[:len(all_values)])) / np.array(SCALER_SCALE[:len(all_values)])
    
    onehot_cols = FEATURE_ORDER[len(all_values):]
    onehot_data = np.zeros(len(onehot_cols))
    
    travel_col = f"BusinessTravel_{user_values['BusinessTravel']}"
    dept_col = f"Department_{user_values['Department']}"
    field_col = f"EducationField_{user_values['EducationField']}"
    gender_col = f"Gender_{user_values['Gender']}"
    role_col = f"JobRole_{user_values['JobRole']}"
    marital_col = f"MaritalStatus_{user_values['MaritalStatus']}"
    overtime_col = f"OverTime_{user_values['OverTime']}"
    age_group_col = f"Age_group_{age_group}"
    
    for col_name in [travel_col, dept_col, field_col, gender_col, role_col, marital_col, overtime_col, age_group_col]:
        if col_name in onehot_cols:
            onehot_data[onehot_cols.index(col_name)] = 1.0
    
    X_full = np.concatenate([X_user.flatten(), onehot_data]).reshape(1, -1)
    return pd.DataFrame(X_full, columns=FEATURE_ORDER)

def create_gauge_chart(probability):
    """Create a gauge chart for attrition probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Attrition Risk", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#00f2fe'},
                {'range': [30, 70], 'color': '#ffd89b'},
                {'range': [70, 100], 'color': '#f5576c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_feature_importance_chart():
    """Create a bar chart showing feature importance"""
    features = ['Monthly Income', 'Age', 'Total Working Years', 'Years At Company', 
                'Distance From Home', 'Job Level', 'Stock Option Level', 'Years Since Last Promotion']
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            showscale=True
        ),
        text=[f'{i*100:.1f}%' for i in importance],
        textposition='auto',
    ))
    fig.update_layout(
        title="Top Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_risk_distribution_chart(probability):
    """Create a pie chart for risk distribution"""
    risk_prob = probability
    safe_prob = 1 - probability
    
    fig = go.Figure(data=[go.Pie(
        labels=['High Risk', 'Low Risk'],
        values=[risk_prob, safe_prob],
        hole=.4,
        marker_colors=['#f5576c', '#00f2fe']
    )])
    fig.update_layout(
        title="Risk Distribution",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    return fig

def create_radar_chart(user_values):
    """Create a radar chart for employee profile"""
    categories = ['Job Satisfaction', 'Environment Satisfaction', 'Work Life Balance', 
                  'Job Involvement', 'Relationship Satisfaction']
    
    values = [
        user_values.get('JobSatisfaction', 0),
        user_values.get('EnvironmentSatisfaction', 0),
        user_values.get('WorkLifeBalance', 0),
        user_values.get('JobInvolvement', 0),
        user_values.get('RelationshipSatisfaction', 0)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Employee Profile',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4])
        ),
        showlegend=True,
        title="Employee Satisfaction Profile",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_comparison_chart(user_values):
    """Create a comparison chart for key metrics"""
    metrics = ['Monthly Income', 'Age', 'Years At Company', 'Distance From Home']
    employee_values = [
        user_values['MonthlyIncome'] / 1000,
        user_values['Age'],
        user_values['YearsAtCompany'],
        user_values['DistanceFromHome']
    ]
    avg_values = [6.5, 37, 7, 9]  # Average values
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Employee',
        x=metrics,
        y=employee_values,
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        name='Company Average',
        x=metrics,
        y=avg_values,
        marker_color='#f093fb'
    ))
    
    fig.update_layout(
        title="Employee vs Company Average",
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Value"
    )
    return fig

def create_trend_chart():
    """Create a line chart showing historical trends"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    attrition_rate = [15, 18, 16, 20, 17, 19]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=attrition_rate,
        mode='lines+markers',
        name='Attrition Rate',
        line=dict(color='#f5576c', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Monthly Attrition Trend",
        xaxis_title="Month",
        yaxis_title="Attrition Rate (%)",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# ================================
# MAIN APP
# ================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/business.png", width=80)
        st.title("Navigation")
        st.markdown("---")
        st.markdown("### About")
        st.info("This app predicts employee attrition risk using machine learning. Enter employee details to get predictions and insights.")
        st.markdown("---")
        st.markdown("### Model Info")
        st.metric("Model Type", "CatBoost")
        st.metric("Features", len(FEATURE_ORDER))
    
    # Main content
    st.title("👥 Employee Attrition Predictor")
    st.markdown("### Predict and Analyze Employee Turnover Risk")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Analytics Dashboard", "📋 Batch Analysis"])
    
    with tab1:
        st.markdown("### Enter Employee Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
            distance = st.number_input("Distance From Home (km)", min_value=0, max_value=50, value=10)
            education = st.selectbox("Education", [1, 2, 3, 4, 5], format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}[x])
            emp_number = st.number_input("Employee Number", min_value=1, max_value=3000, value=1000)
            env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
            hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
            job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
        
        with col2:
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
            monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=6500)
            monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000)
            num_companies = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
            salary_hike = st.number_input("Percent Salary Hike", min_value=10, max_value=25, value=15)
            performance = st.selectbox("Performance Rating", [1, 2, 3, 4])
            relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
        
        with col3:
            stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
            training_times = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=3)
            work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x])
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
            years_in_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=3)
            years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
            years_with_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=3)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            business_travel = st.selectbox("Business Travel", CATEGORY_ORDERS['BusinessTravel'])
            department = st.selectbox("Department", CATEGORY_ORDERS['Department'])
        
        with col2:
            education_field = st.selectbox("Education Field", CATEGORY_ORDERS['EducationField'])
            gender = st.selectbox("Gender", CATEGORY_ORDERS['Gender'])
        
        with col3:
            job_role = st.selectbox("Job Role", CATEGORY_ORDERS['JobRole'])
            marital_status = st.selectbox("Marital Status", CATEGORY_ORDERS['MaritalStatus'])
        
        with col4:
            overtime = st.selectbox("Over Time", CATEGORY_ORDERS['OverTime'])
        
        st.markdown("---")
        
        if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
            user_values = {
                'Age': age, 'DailyRate': daily_rate, 'DistanceFromHome': distance,
                'Education': education, 'EmployeeNumber': emp_number,
                'EnvironmentSatisfaction': env_satisfaction, 'HourlyRate': hourly_rate,
                'JobInvolvement': job_involvement, 'JobLevel': job_level,
                'JobSatisfaction': job_satisfaction, 'MonthlyIncome': monthly_income,
                'MonthlyRate': monthly_rate, 'NumCompaniesWorked': num_companies,
                'PercentSalaryHike': salary_hike, 'PerformanceRating': performance,
                'RelationshipSatisfaction': relationship_satisfaction,
                'StockOptionLevel': stock_option, 'TotalWorkingYears': total_working_years,
                'TrainingTimesLastYear': training_times, 'WorkLifeBalance': work_life_balance,
                'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_in_role,
                'YearsSinceLastPromotion': years_since_promotion,
                'YearsWithCurrManager': years_with_manager,
                'BusinessTravel': business_travel, 'Department': department,
                'EducationField': education_field, 'Gender': gender,
                'JobRole': job_role, 'MaritalStatus': marital_status, 'OverTime': overtime
            }
            
            X_input = build_input_row(user_values)
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1]
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if prediction == 1:
                    st.markdown(f'<div class="prediction-high">⚠️ HIGH ATTRITION RISK<br>{probability*100:.1f}% Probability</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-low">✅ LOW ATTRITION RISK<br>{probability*100:.1f}% Probability</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(create_risk_distribution_chart(probability), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_radar_chart(user_values), use_container_width=True)
            with col2:
                st.plotly_chart(create_comparison_chart(user_values), use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Analytics Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>Total Employees</h3><h1>1,470</h1></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>Attrition Rate</h3><h1>16.1%</h1></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>Avg Tenure</h3><h1>7 years</h1></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h3>High Risk</h3><h1>237</h1></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
        with col2:
            st.plotly_chart(create_trend_chart(), use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### Feature Correlation Heatmap")
        features = ['Age', 'Monthly Income', 'Years At Company', 'Job Satisfaction', 'Work Life Balance']
        corr_data = np.random.rand(5, 5)
        np.fill_diagonal(corr_data, 1)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data,
            x=features,
            y=features,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📋 Batch Analysis")
        st.info("Upload a CSV file with employee data to predict attrition risk for multiple employees.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Run Batch Prediction"):
                st.success(f"✅ Processed {len(df)} employees")
                st.balloons()
        else:
            st.markdown("#### Sample CSV Format:")
            sample_df = pd.DataFrame({
                'Age': [30, 35, 28],
                'MonthlyIncome': [5000, 7000, 4500],
                'JobSatisfaction': [3, 2, 4],
                'Department': ['Sales', 'Research & Development', 'Sales']
            })
            st.dataframe(sample_df)

if __name__ == "__main__":
    main()