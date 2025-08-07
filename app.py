import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import statsmodels.api as sm
from textblob import TextBlob

st.set_page_config(page_title="Dubai AI Shopping Assistant Dashboard", layout="wide")

# Upload data
st.title("Dubai AI Shopping Assistant Survey Dashboard")
st.markdown("_Upload your latest Google Forms Excel/CSV to update the dashboard_")

uploaded_file = st.file_uploader("Upload survey data (.csv or .xlsx)", type=["csv", "xlsx"])

if not uploaded_file:
    st.info("Please upload your exported Google Forms file to begin.")
    st.stop()

if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# Clean columns
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
    'I find AI shopping assistants easy to use and understand.',
    'It doesn’t take much effort to learn how to use these assistants.',
    'AI shopping assistants give me recommendations that match my taste.',
    'I feel like AI shopping assistants understand what I want.',
    'I appreciate when an AI shopping assistant speaks my language or uses familiar cultural references.',
    'I enjoy chatting with an AI shopping assistant.',
    'Sometimes, AI assistants “get me” better than human staff.'
]

available_likert_cols = [col for col in likert_cols if col in filtered_df.columns]
if not available_likert_cols:
    st.warning("No Likert-scale questions available in filtered data for analysis.")
    st.stop()

mapped_scores = filtered_df[available_likert_cols].replace(likert_map)
filtered_df['Avg Likert Score'] = mapped_scores.mean(axis=1)
avg_likert_score = filtered_df['Avg Likert Score'].mean()

# Define tabs
tabs = st.tabs([
    "Overview & Demographics",
    "Likert Analysis & Correlations",
    "Regression & What-if",
    "Sentiment Analysis",
    "KPIs & Recommendations"
])

# 1. Overview & Demographics
with tabs[0]:
    st.subheader("Quick Demographics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responses", len(filtered_df))
    col2.metric("Avg Digital Comfort", round(filtered_df['How comfortable are you with using new digital technology?'].mean(), 2))
    col3.metric("Online Shoppers (%)", round((filtered_df['How do you usually shop in Dubai?'].value_counts(normalize=True).get('Online (web/app)', 0)) * 100, 1))
    col4.metric("Avg Likert Score", round(avg_likert_score, 2))

    st.markdown("---")
    st.subheader("Age and Nationality Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    filtered_df['How old are you?'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='royalblue', title="Age")
    filtered_df['What best describes you?'].value_counts().plot(kind='pie', autopct='%1.0f%%', ax=ax[1], title="Nationality")
    st.pyplot(fig)

# 2. Likert Analysis & Correlations
with tabs[1]:
    st.subheader("Likert Questions Average Scores")
    means = mapped_scores.mean()
    st.bar_chart(means)

    st.subheader("Average Scores for Each Likert Question")
    avg_scores_df = pd.DataFrame({
        'Question': means.index,
        'Average Score': means.values.round(2)
    })
    st.table(avg_scores_df)

    st.subheader("Correlation Heatmap of Attitudes")
    corr = mapped_scores.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# 3. Regression & What-if
with tabs[2]:
    st.subheader("Regression: Drivers of Emotional Engagement")
    target_col = 'Sometimes, AI assistants “get me” better than human staff.'
    if target_col in mapped_scores.columns:
        y = mapped_scores[target_col]
        X = mapped_scores.drop(columns=[target_col])
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

        # Highlight only the significant one
        st.markdown("**Key Finding:**")
        sig = results_df.loc['I enjoy chatting with an AI shopping assistant.']
        st.markdown(
            f"- **Enjoyment** (`I enjoy chatting with an AI shopping assistant`) is the **only statistically significant driver** (β = {sig['Coefficient']:.2f}, p = {sig['P-value']:.3f}) of emotional engagement.\n"
            f"- Other factors (usefulness, trust, ease, privacy, culture) are positive but not significant after accounting for enjoyment."
        )

        st.markdown("> **Managerial Implication:** To boost engagement, make AI more enjoyable, interactive, and emotionally rewarding.")

        # What-if analysis: bar chart
        st.subheader("What-If: Engagement Lift per +1 Point Increase")
        bar_vals = results_df['Coefficient'].drop('const')
        colors = ['#38b6ff' if idx == 'I enjoy chatting with an AI shopping assistant.' else '#d1d5db' for idx in bar_vals.index]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(bar_vals.index, bar_vals.values, color=colors, edgecolor='black')
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel("Engagement Lift")
        st.pyplot(fig)

        st.caption("Note: Negative coefficients do not mean these features are unimportant, but only enjoyment uniquely drives engagement in this model.")
    else:
        st.warning(f"Column '{target_col}' not found in data.")

# 4. Sentiment Analysis
with tabs[3]:
    st.subheader("Sentiment Analysis of Open-Ended Feedback")
    # Automatically find all open-ended columns (you can adjust range as needed)
    open_ended_cols = [col for col in df.columns if df[col].dtype == object and df[col].nunique() > 8 and col not in likert_cols]
    if open_ended_cols:
        all_comments = filtered_df[open_ended_cols].astype(str).agg(' '.join, axis=1)
        sentiments = all_comments.apply(lambda x: TextBlob(x).sentiment.polarity)
        sentiment_label = sentiments.apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
        sentiment_numeric = sentiment_label.map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

        pos_pct = (sentiment_label == "Positive").mean() * 100
        neut_pct = (sentiment_label == "Neutral").mean() * 100
        neg_pct = (sentiment_label == "Negative").mean() * 100
        avg_sentiment_score = sentiment_numeric.mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Sentiment Score", f"{avg_sentiment_score:.2f}")
            st.metric("Positive (%)", f"{pos_pct:.0f}%")
            st.metric("Neutral (%)", f"{neut_pct:.0f}%")
            st.metric("Negative (%)", f"{neg_pct:.0f}%")

        # Pie chart with fun emojis
        pie_labels = [f"😊 Positive", f"😐 Neutral", f"😞 Negative"]
        pie_vals = [pos_pct, neut_pct, neg_pct]
        pie_colors = ['#62b5e5', '#f7d283', '#e86c5c']
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(pie_vals, labels=pie_labels, colors=pie_colors, autopct='%1.0f%%', startangle=120, wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
        ax.set_title("Sentiment Breakdown")
        st.pyplot(fig)

        # Word cloud
        st.subheader("Word Cloud – Shopper Feedback")
        wc = WordCloud(width=900, height=400, background_color='white', colormap='winter', max_words=50).generate(' '.join(all_comments))
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Sample comments table
        st.markdown("#### Example Shopper Quotes")
        for sent in ['Positive', 'Neutral', 'Negative']:
            sample = all_comments[sentiment_label == sent]
            if not sample.empty:
                st.write(f"**{sent}:** _{sample.iloc[0][:140]}..._")

        st.markdown(
            "> **Actionable Insights:** Expand language/cultural support, clarify privacy, add more fun features, enhance personalization."
        )
    else:
        st.info("No open-ended feedback columns detected.")

# 5. KPIs & Recommendations
with tabs[4]:
    st.subheader("Key Performance Indicators & Recommendations")

    st.metric("Total Survey Responses", len(filtered_df))
    st.metric("Average Digital Comfort Score", round(filtered_df['How comfortable are you with using new digital technology?'].mean(), 2))
    st.metric("Average Likert Score", round(avg_likert_score, 2))

    target_col_recommend = 'Would you recommend using AI-powered shopping assistants to others?'
    if target_col_recommend in filtered_df.columns:
        adoption_rate = filtered_df[target_col_recommend].map({'Yes': 1, 'No': 0, 'Maybe': 0}).mean()
        st.metric("Adoption Rate (Recommend AI Assistants)", f"{adoption_rate:.2%}")
    else:
        adoption_rate = None

    barriers_col = 'Are there any concerns or turn-offs for you with AI in shopping? '
    if barriers_col in filtered_df.columns:
        barrier_counts = filtered_df[barriers_col].value_counts()
        st.write("Top Reported Barriers:")
        st.bar_chart(barrier_counts)

    st.markdown("**Strategic Recommendations:**")
st.write("- Prioritize building trust with clear privacy controls and transparency.")
st.write("- Enhance personalization: Use AI to deliver smarter, more relevant deals and offers.")
st.write("- Make the AI assistant more enjoyable—add interactive features and rewards.")
st.write("- Expand language/cultural support to include more shopper segments.")
st.write("- Provide easy access to human staff for complex queries.")
st.write("- Continuously collect feedback and iterate the experience.")


st.markdown("---")
st.caption("Dashboard by Sanchit Singh Thapa | MBA Research | SP Jain")
