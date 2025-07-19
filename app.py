import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import statsmodels.api as sm

st.title("Dubai AI Shopping Assistant Survey Dashboard")
st.markdown("_Upload your latest Google Forms Excel/CSV to update the dashboard_")

uploaded_file = st.file_uploader("Upload your survey data file here (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"Loaded {len(df)} responses! Ready for analysis.")
else:
    st.info("Please upload your exported Google Forms file to begin.")
    st.stop()

# Data Cleaning - strip whitespace from column names
df.columns = df.columns.str.strip()

# Sidebar Filters
with st.sidebar:
    st.header("Filter Responses")
    age_filter = st.multiselect("Age", options=df['How old are you?'].unique())
    nationality_filter = st.multiselect("Nationality", options=df['What best describes you?'].unique())
    shopping_style_filter = st.multiselect("Shopping Style", options=df['How do you usually shop in Dubai?'].unique())

filtered_df = df.copy()
if age_filter:
    filtered_df = filtered_df[filtered_df['How old are you?'].isin(age_filter)]
if nationality_filter:
    filtered_df = filtered_df[filtered_df['What best describes you?'].isin(nationality_filter)]
if shopping_style_filter:
    filtered_df = filtered_df[filtered_df['How do you usually shop in Dubai?'].isin(shopping_style_filter)]

# Likert mapping
likert_map = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5
}

likert_cols = [
    'AI assistants help me make better shopping decisions.',
    'AI assistants save me time in-store or online.',
    'I trust recommendations made by AI shopping assistants.',
    'I feel confident that my data is safe when using AI features.',
    'I feel confident that my data is safe when using AI features..1',
    'I find AI shopping assistants easy to use and understand.',
    'It doesn’t take much effort to learn how to use these assistants.',
    'AI shopping assistants give me recommendations that match my taste.',
    'I feel like AI shopping assistants understand what I want.',
    'I appreciate when an AI shopping assistant speaks my language or uses familiar cultural references.',
    'I enjoy chatting with an AI shopping assistant.',
    'Sometimes, AI assistants “get me” better than human staff.'
]

# ========== KEY CHANGE: Filter likert_cols dynamically =============
available_likert_cols = [col for col in likert_cols if col in filtered_df.columns]
if not available_likert_cols:
    st.warning("No Likert-scale questions available in filtered data for analysis.")
    st.stop()
# ===================================================================

mapped_scores = filtered_df[available_likert_cols].replace(likert_map)

# Calculate Avg Likert Score per respondent and overall
filtered_df['Avg Likert Score'] = mapped_scores.mean(axis=1)
avg_likert_score = filtered_df['Avg Likert Score'].mean()

# Metrics display
st.subheader("Quick Demographics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Responses", len(filtered_df))
col2.metric("Avg Digital Comfort", round(filtered_df['How comfortable are you with using new digital technology?'].mean(), 2))
col3.metric("Online Shoppers (%)", round((filtered_df['How do you usually shop in Dubai?'].value_counts(normalize=True).get('Online (web/app)', 0)) * 100, 1))
col4.metric("Avg Likert Score", round(avg_likert_score, 2))

st.markdown("---")

# Respondent Profile Visualization
st.subheader("Respondent Profile")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
filtered_df['How old are you?'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='royalblue', title="Age")
filtered_df['What best describes you?'].value_counts().plot(kind='pie', autopct='%1.0f%%', ax=ax[1], title="Nationality")
st.pyplot(fig)

# Likert means bar chart
st.subheader("Key Drivers: Means (1=Strongly Disagree, 5=Strongly Agree)")
means = mapped_scores.mean()
st.bar_chart(means)

# Correlation Heatmap
st.subheader("Correlation Heatmap of Attitudes")
corr = mapped_scores.corr()
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Regression
st.subheader("Regression: What drives positive response to AI shopping assistants?")
target_col = 'Imagine you’re at Dubai Mall and a smart screen offers you a personalised deal based on your preferences, and can even speak your language.'

if target_col in filtered_df.columns:
    response_map = {
        'Excited': 5,
        'Curious but cautious': 4,
        'Neutral': 3,
        'Uncomfortable': 2,
        'Annoyed': 1
    }
    y = filtered_df[target_col].map(response_map)
    X = mapped_scores
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()

    results_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std Error': model.bse,
        'P-value': model.pvalues
    })

    st.dataframe(results_df.style.format({
        'Coefficient': '{:.3f}',
        'Std Error': '{:.3f}',
        'P-value': '{:.3f}'
    }))

    significant = results_df[results_df['P-value'] < 0.05]
    if not significant.empty:
        st.markdown("**Significant Predictors (p < 0.05):**")
        st.dataframe(significant)

    st.markdown("""
    > **Interpretation:**  
    > Predictors with positive coefficients and p-values below 0.05 significantly influence positive reactions to AI shopping assistants.
    """)
else:
    st.warning(f"Column '{target_col}' not found in data.")

# Word Cloud for open-ended feedback
st.subheader("Open-Ended Feedback (Word Cloud)")
open_ended_col = 'Any ideas or suggestions for how Dubai retailers can make AI shopping assistants better for you?'

if open_ended_col in filtered_df.columns:
    text = ' '.join(filtered_df[open_ended_col].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

st.markdown("---")
st.caption("Dashboard by Sanchit Singh Thapa | MBA Research | SP Jain")
