import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import sqlite3
from uuid import uuid4
import io

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            Employee_ID INTEGER,
            Department TEXT,
            Training_Program TEXT,
            Pre_Training_Score REAL,
            Post_Training_Score REAL,
            Feedback TEXT,
            Engagement_hrs REAL,
            Sentiment_Score REAL,
            Manager_Support_Rating INTEGER,
            Learning_Style TEXT,
            Training_Difficulty TEXT,
            Trainer_Quality INTEGER,
            Engagement_Support_Score REAL,
            Score_Progress_Indicator REAL,
            Training_Effectiveness REAL,
            Motivation_Index REAL,
            High_Performance INTEGER,
            Improvement_pct REAL,
            Computed_Sentiment REAL
        )
    ''')
    conn.commit()
    return conn

# --- Data Loading and Cleaning ---
def load_and_clean_data(file_content):
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Clean data
    df = df.dropna()  # Remove rows with missing values
    df = df[df['Pre_Training_Score'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
    df = df[df['Post_Training_Score'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
    df['Feedback'] = df['Feedback'].astype(str).apply(lambda x: x.strip().replace('"', ''))

    # Compute sentiment globally
    df['Computed_Sentiment'] = df['Feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Define expected columns
    numeric_cols = ['Pre_Training_Score', 'Post_Training_Score', 'Engagement (hrs)', 'Sentiment_Score',
                    'Manager_Support_Rating', 'Trainer_Quality', 'Engagement_Support_Score',
                    'Score_Progress_Indicator', 'Training_Effectiveness', 'Motivation_Index', 
                    'Improvement (%)', 'Computed_Sentiment']
    
    # Check for missing columns
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
        return None

    # Ensure numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    return df

# --- Store Data in SQLite ---
def store_data(df, conn):
    df.to_sql('training_data', conn, if_exists='replace', index=False)

# --- Sentiment Analysis ---
def analyze_feedback_sentiment(feedback):
    analysis = TextBlob(feedback)
    return analysis.sentiment.polarity

# --- Training Effectiveness Analysis ---
def calculate_effectiveness(df):
    df['Score_Improvement'] = df['Post_Training_Score'] - df['Pre_Training_Score']
    effectiveness_summary = df.groupby('Training_Program').agg({
        'Score_Improvement': 'mean',
        'Post_Training_Score': 'mean',
        'Engagement (hrs)': 'mean',
        'Sentiment_Score': 'mean',
        'Computed_Sentiment': 'mean'
    }).reset_index()
    return effectiveness_summary

# --- Clustering Employees ---
def cluster_employees(df):
    features = df[['Pre_Training_Score', 'Post_Training_Score', 'Engagement (hrs)', 'Sentiment_Score', 'Computed_Sentiment']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    cluster_summary = df.groupby('Cluster').agg({
        'Pre_Training_Score': 'mean',
        'Post_Training_Score': 'mean',
        'Engagement (hrs)': 'mean',
        'Sentiment_Score': 'mean',
        'Computed_Sentiment': 'mean'
    }).reset_index()
    return df, cluster_summary

# --- Regression Analysis ---
def predict_outcomes(df):
    X = df[['Pre_Training_Score', 'Engagement (hrs)', 'Manager_Support_Rating', 'Trainer_Quality']].values
    y = df['Post_Training_Score'].values
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    df['Predicted_Post_Score'] = predictions
    return df, model

# --- Single Prediction ---
def predict_single(model, pre_score, engagement, manager_support, trainer_quality):
    input_data = np.array([[pre_score, engagement, manager_support, trainer_quality]])
    prediction = model.predict(input_data)[0]
    return prediction

# --- Recommendations ---
def generate_recommendations(effectiveness_summary, cluster_summary, department=None, program=None, cluster=None):
    recommendations = []
    
    # Filter data if specified
    eff_summary = effectiveness_summary
    cl_summary = cluster_summary
    
    if program:
        eff_summary = eff_summary[eff_summary['Training_Program'] == program]
    
    # Top and bottom performing programs
    if not eff_summary.empty:
        top_programs = eff_summary.nlargest(2, 'Score_Improvement')['Training_Program'].tolist()
        bottom_programs = eff_summary.nsmallest(2, 'Score_Improvement')['Training_Program'].tolist()
        recommendations.append(f"**Top Performing Programs**: {', '.join(top_programs) if top_programs else 'None'}. Consider scaling these programs.")
        recommendations.append(f"**Programs Needing Improvement**: {', '.join(bottom_programs) if bottom_programs else 'None'}. Review content and delivery methods.")
    
    # Cluster-based recommendations
    if cluster is not None:
        cl_summary = cl_summary[cl_summary['Cluster'] == cluster]
    
    for idx, row in cl_summary.iterrows():
        if row['Post_Training_Score'] < 70:
            recommendations.append(f"Cluster {int(row['Cluster'])}: Low post-training scores ({row['Post_Training_Score']:.2f}). Increase engagement and support.")
        elif row['Engagement (hrs)'] < 10:
            recommendations.append(f"Cluster {int(row['Cluster'])}: Low engagement ({row['Engagement (hrs)']:.2f} hrs). Introduce interactive elements.")
    
    if department:
        recommendations.append(f"**Department-Specific ({department})**: Analyze department-specific engagement and tailor training formats.")
    
    return recommendations

# --- Visualizations ---
def plot_score_improvement(df, department=None, program=None):
    filtered_df = df
    if department:
        filtered_df = filtered_df[filtered_df['Department'] == department]
    if program:
        filtered_df = filtered_df[filtered_df['Training_Program'] == program]
    fig = px.histogram(filtered_df, x='Improvement (%)', nbins=50, title='Distribution of Score Improvement (%)',
                       labels={'Improvement (%)': 'Score Improvement (%)'},
                       color_discrete_sequence=['#636EFA'])
    fig.update_layout(xaxis_title="Score Improvement (%)", yaxis_title="Count", showlegend=False)
    return fig

def plot_program_effectiveness(effectiveness_summary, department=None):
    fig = px.bar(effectiveness_summary, x='Training_Program', y='Score_Improvement',
                 title='Average Score Improvement by Training Program',
                 labels={'Score_Improvement': 'Average Score Improvement'},
                 color='Score_Improvement', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title="Training Program", yaxis_title="Average Score Improvement")
    return fig

def plot_engagement_metrics(df, department=None, program=None):
    filtered_df = df
    if department:
        filtered_df = filtered_df[filtered_df['Department'] == department]
    if program:
        filtered_df = filtered_df[filtered_df['Training_Program'] == program]
    fig = px.scatter(filtered_df, x='Engagement (hrs)', y='Post_Training_Score', color='Training_Program',
                     title='Engagement vs. Post-Training Score',
                     labels={'Engagement (hrs)': 'Engagement (Hours)', 'Post_Training_Score': 'Post-Training Score'})
    fig.update_layout(xaxis_title="Engagement (Hours)", yaxis_title="Post-Training Score")
    return fig

def plot_cluster_analysis(df, department=None):
    filtered_df = df
    if department:
        filtered_df = filtered_df[filtered_df['Department'] == department]
    fig = px.scatter(filtered_df, x='Pre_Training_Score', y='Post_Training_Score', color='Cluster',
                     title='Employee Clusters Based on Performance',
                     labels={'Pre_Training_Score': 'Pre-Training Score', 'Post_Training_Score': 'Post-Training Score'})
    fig.update_layout(xaxis_title="Pre-Training Score", yaxis_title="Post-Training Score")
    return fig

def plot_sentiment_analysis(df, department=None, program=None):
    filtered_df = df
    if department:
        filtered_df = filtered_df[filtered_df['Department'] == department]
    if program:
        filtered_df = filtered_df[filtered_df['Training_Program'] == program]
    fig = px.box(filtered_df, x='Training_Program', y='Computed_Sentiment', 
                 title='Sentiment Analysis of Feedback by Program',
                 labels={'Computed_Sentiment': 'Sentiment Score (-1 to 1)'},
                 color='Training_Program')
    fig.update_layout(xaxis_title="Training Program", yaxis_title="Sentiment Score (-1 to 1)")
    return fig

# --- Streamlit App ---
st.set_page_config(page_title="Employee Training Effectiveness Analyzer", layout="wide")
st.title("Employee Training Effectiveness Analyzer")
st.markdown("Interactively analyze training program effectiveness, predict performance, and generate tailored recommendations.")

# Initialize database
conn = init_db()

# Sidebar: Data Input and Interactive Tools
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'regression_model' not in st.session_state:
    st.session_state.regression_model = None

if uploaded_file:
    file_content = uploaded_file.read()
    df = load_and_clean_data(file_content)
    
    if df is not None:
        # Store data in SQLite
        store_data(df, conn)
        st.session_state.df = df
        
        # Perform Analyses
        effectiveness_summary = calculate_effectiveness(df)
        df, cluster_summary = cluster_employees(df)
        df, regression_model = predict_outcomes(df)
        st.session_state.regression_model = regression_model
        st.session_state.df = df  # Update with new columns
        
        # Sidebar: Prediction Tool
        st.sidebar.header("Predict Performance")
        pre_score = st.sidebar.slider("Pre-Training Score", 0.0, 100.0, 50.0)
        engagement = st.sidebar.slider("Engagement (hrs)", 0.0, 50.0, 10.0)
        manager_support = st.sidebar.slider("Manager Support Rating", 1, 5, 3)
        trainer_quality = st.sidebar.slider("Trainer Quality", 1, 5, 3)
        
        if st.sidebar.button("Predict Post-Training Score"):
            prediction = predict_single(regression_model, pre_score, engagement, manager_support, trainer_quality)
            st.sidebar.success(f"Predicted Post-Training Score: {prediction:.2f}")
        
        # Sidebar: Employee Lookup
        st.sidebar.header("Employee Lookup")
        employee_id = st.sidebar.number_input("Enter Employee ID", min_value=1, step=1)
        if st.sidebar.button("Search Employee"):
            employee_data = df[df['Employee_ID'] == employee_id]
            if not employee_data.empty:
                st.sidebar.write("**Employee Details**")
                st.sidebar.write(f"Department: {employee_data['Department'].iloc[0]}")
                st.sidebar.write(f"Training Program: {employee_data['Training_Program'].iloc[0]}")
                st.sidebar.write(f"Pre-Training Score: {employee_data['Pre_Training_Score'].iloc[0]:.2f}")
                st.sidebar.write(f"Post-Training Score: {employee_data['Post_Training_Score'].iloc[0]:.2f}")
                st.sidebar.write(f"Predicted Post-Training Score: {employee_data['Predicted_Post_Score'].iloc[0]:.2f}")
                st.sidebar.write(f"Feedback: {employee_data['Feedback'].iloc[0]}")
                st.sidebar.write(f"Sentiment Score: {employee_data['Computed_Sentiment'].iloc[0]:.2f}")
                st.sidebar.write(f"Cluster: {int(employee_data['Cluster'].iloc[0])}")
            else:
                st.sidebar.error("Employee ID not found.")
        
        # Sidebar: Feedback Submission
        st.sidebar.header("Submit Feedback")
        feedback_employee_id = st.sidebar.number_input("Employee ID for Feedback", min_value=1, step=1, key="feedback_id")
        new_feedback = st.sidebar.text_area("Enter Feedback")
        if st.sidebar.button("Submit Feedback"):
            if new_feedback and feedback_employee_id in df['Employee_ID'].values:
                sentiment = analyze_feedback_sentiment(new_feedback)
                df.loc[df['Employee_ID'] == feedback_employee_id, 'Feedback'] = new_feedback
                df.loc[df['Employee_ID'] == feedback_employee_id, 'Sentiment_Score'] = sentiment
                df.loc[df['Employee_ID'] == feedback_employee_id, 'Computed_Sentiment'] = sentiment
                store_data(df, conn)
                st.session_state.df = df
                st.sidebar.success("Feedback updated successfully!")
            else:
                st.sidebar.error("Invalid Employee ID or feedback.")
        
        # Filters for Visualizations and Recommendations
        departments = ['All'] + sorted(df['Department'].unique().tolist())
        programs = ['All'] + sorted(df['Training_Program'].unique().tolist())
        clusters = ['All'] + sorted(df['Cluster'].unique().tolist())
        
        st.sidebar.header("Filters")
        selected_department = st.sidebar.selectbox("Department", departments)
        selected_program = st.sidebar.selectbox("Training Program", programs)
        selected_cluster = st.sidebar.selectbox("Cluster", clusters, format_func=lambda x: 'All' if x == 'All' else f"Cluster {int(x)}")
        
        # Apply filters
        filtered_df = df
        filtered_effectiveness = effectiveness_summary
        filtered_cluster_summary = cluster_summary
        
        if selected_department != 'All':
            filtered_df = filtered_df[filtered_df['Department'] == selected_department]
            filtered_effectiveness = filtered_df.groupby('Training_Program').agg({
                'Score_Improvement': 'mean',
                'Post_Training_Score': 'mean',
                'Engagement (hrs)': 'mean',
                'Sentiment_Score': 'mean',
                'Computed_Sentiment': 'mean'
            }).reset_index()
            filtered_cluster_summary = filtered_df.groupby('Cluster').agg({
                'Pre_Training_Score': 'mean',
                'Post_Training_Score': 'mean',
                'Engagement (hrs)': 'mean',
                'Sentiment_Score': 'mean',
                'Computed_Sentiment': 'mean'
            }).reset_index()
        
        if selected_program != 'All':
            filtered_df = filtered_df[filtered_df['Training_Program'] == selected_program]
        
        if selected_cluster != 'All':
            filtered_df = filtered_df[filtered_df['Cluster'] == selected_cluster]
        
        # Generate recommendations with filters
        recommendations = generate_recommendations(
            filtered_effectiveness,
            filtered_cluster_summary,
            department=selected_department if selected_department != 'All' else None,
            program=selected_program if selected_program != 'All' else None,
            cluster=int(selected_cluster) if selected_cluster != 'All' else None
        )
        
        # Tabs for Different Sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Effectiveness", "Engagement", "Feedback", "Recommendations"])
        
        with tab1:
            st.header("Overview")
            st.write("Summary of training data and key metrics.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Employees", len(filtered_df))
            with col2:
                st.metric("Average Improvement", f"{filtered_df['Improvement (%)'].mean():.2f}%")
            with col3:
                st.metric("Average Engagement", f"{filtered_df['Engagement (hrs)'].mean():.2f} hrs")
            st.plotly_chart(plot_score_improvement(filtered_df, selected_department, selected_program), use_container_width=True)
        
        with tab2:
            st.header("Training Effectiveness")
            st.write("Compare pre- and post-training performance metrics.")
            st.plotly_chart(plot_program_effectiveness(filtered_effectiveness, selected_department), use_container_width=True)
            st.dataframe(filtered_effectiveness.style.format({
                "Score_Improvement": "{:.2f}",
                "Post_Training_Score": "{:.2f}",
                "Engagement (hrs)": "{:.2f}",
                "Sentiment_Score": "{:.2f}",
                "Computed_Sentiment": "{:.2f}"
            }))
        
        with tab3:
            st.header("Engagement Metrics")
            st.write("Visualize employee participation and engagement levels.")
            st.plotly_chart(plot_engagement_metrics(filtered_df, selected_department, selected_program), use_container_width=True)
            st.plotly_chart(plot_cluster_analysis(filtered_df, selected_department), use_container_width=True)
            st.dataframe(filtered_cluster_summary.style.format({
                'Pre_Training_Score': '{:.2f}',
                'Post_Training_Score': '{:.2f}',
                'Engagement (hrs)': '{:.2f}',
                'Sentiment_Score': '{:.2f}',
                'Computed_Sentiment': '{:.2f}'
            }))
        
        with tab4:
            st.header("Feedback Analysis")
            st.write("Sentiment analysis of employee feedback.")
            st.plotly_chart(plot_sentiment_analysis(filtered_df, selected_department, selected_program), use_container_width=True)
            st.write("Sample Feedback:")
            st.write(filtered_df[['Employee_ID', 'Feedback', 'Computed_Sentiment']].head(10))
        
        with tab5:
            st.header("Recommendations")
            st.write("Actionable insights to improve training programs.")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Export Analysis
        st.sidebar.header("Export Analysis")
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Processed Data",
            data=csv,
            file_name="processed_training_data.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file to begin analysis.")

# Close database connection
conn.close()