import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
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

# Define open-ended feedback column here near top
open_ended_col = 'Any ideas or suggestions for how Dubai retailers can make AI shopping assistants better for you?'

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
    'I feel confident that my data is safe when using AI features..1',
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
    "Regression & Prediction",
    "Clustering",
    "Customer Journey",
    "Sentiment & Topic Modeling",
    "Scenario Simulations",
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

# 3. Regression & Prediction
with tabs[2]:
    st.subheader("Regression: Positive Reaction to AI Assistants")

    target_col_pos_reaction = 'Imagine you’re at Dubai Mall and a smart screen offers you a personalised deal based on your preferences, and can even speak your language.'

    if target_col_pos_reaction in filtered_df.columns:
        response_map = {
            'Excited': 5,
            'Curious but cautious': 4,
            'Neutral': 3,
            'Uncomfortable': 2,
            'Annoyed': 1
        }
        y = filtered_df[target_col_pos_reaction].map(response_map)
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
        st.warning(f"Column '{target_col_pos_reaction}' not found in data.")

    # Random Forest Classification for adoption prediction
    st.subheader("Random Forest: Predict Willingness to Recommend AI Assistants")

    target_col_recommend = 'Would you recommend using AI-powered shopping assistants to others?'

    if target_col_recommend in filtered_df.columns:
        rf_target_map = {'Yes': 1, 'No': 0, 'Maybe': 0}
        y_rf = filtered_df[target_col_recommend].map(rf_target_map).dropna()
        X_rf = mapped_scores.loc[y_rf.index]
        X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.25, random_state=42)

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        feat_imp = pd.Series(rf.feature_importances_, index=X_rf.columns).sort_values(ascending=False)
        st.bar_chart(feat_imp)
    else:
        st.warning(f"Column '{target_col_recommend}' not found in data.")

# 4. Clustering & Shopper Profiles
with tabs[3]:
    st.subheader("K-Means Clustering: Shopper Segmentation")
    n_clusters = st.slider("Select number of clusters", 2, 6, 3)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mapped_scores)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    filtered_df['Cluster'] = clusters

    st.write("Cluster sizes:")
    st.write(pd.Series(clusters).value_counts())

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=mapped_scores.columns)
    st.write("Cluster centers (average scores):")
    st.dataframe(centers)

# 5. Customer Journey (Sankey Diagram)
with tabs[4]:
    st.subheader("Customer Journey Flow: Awareness → Trust → Satisfaction → Adoption")

    labels = [
        "Aware", "Not Aware",
        "Trust High", "Trust Low",
        "Satisfied", "Not Satisfied",
        "Adopted", "Not Adopted"
    ]

    source = [0, 0, 2, 2, 4, 4, 6, 6]
    target = [2, 3, 4, 5, 6, 7, 7, 6]
    value = [100, 50, 80, 20, 70, 10, 60, 5]

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, label=labels),
        link=dict(source=source, target=target, value=value)
    )])

    st.plotly_chart(fig, use_container_width=True)

# 6. Sentiment & Topic Modeling
with tabs[5]:
    st.subheader("Sentiment Analysis of Open-Ended Responses")

    if open_ended_col in filtered_df.columns:
        comments = filtered_df[open_ended_col].dropna().astype(str)

        sentiments = comments.apply(lambda x: TextBlob(x).sentiment.polarity)
        filtered_df.loc[comments.index, 'Sentiment'] = sentiments

        st.write("Sentiment distribution:")
        st.bar_chart(sentiments.value_counts(bins=5, sort=False))

        st.metric("Average Sentiment Polarity", round(sentiments.mean(), 3))

        st.subheader("Topic Modeling")

        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        dtm = vectorizer.fit_transform(comments)
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(dtm)

        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics[f"Topic {idx+1}"] = ", ".join(top_words)

        for topic, words in topics.items():
            st.write(f"**{topic}:** {words}")
    else:
        st.warning("No open-ended feedback found.")

# 7. Scenario-Based Simulations
with tabs[6]:
    st.subheader("Scenario Simulator: Impact of Personalization Score")

    if 'model' in locals():
        personalization_var = 'AI shopping assistants give me recommendations that match my taste.'

        if personalization_var in available_likert_cols:
            coeff = model.params.get(personalization_var, None)
            if coeff is not None:
                slider_val = st.slider("Increase Personalization Score by", 0.0, 2.0, 0.0, 0.1)
                base_avg = filtered_df[personalization_var].replace(likert_map).mean()
                predicted_change = coeff * slider_val
                new_score = base_avg + predicted_change
                st.write(f"Base avg personalization score: {base_avg:.2f}")
                st.write(f"Predicted change in positive reaction: {predicted_change:.3f}")
                st.write(f"New predicted average positive reaction score: {new_score:.3f}")
            else:
                st.info("Personalization variable coefficient not found in regression model.")
        else:
            st.info("Personalization variable not in filtered data.")
    else:
        st.info("Run regression first to use scenario simulator.")

# 8. KPIs & Recommendations
with tabs[7]:
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

    st.markdown("### Recommendations")
    if adoption_rate is not None:
        if adoption_rate < 0.5:
            st.write("- Focus on building trust and transparency about data privacy.")
        else:
            st.write("- Continue enhancing personalization and ease of use.")
        if filtered_df['How comfortable are you with using new digital technology?'].mean() < 3:
            st.write("- Provide educational resources to increase digital comfort.")
        else:
            st.write("- Leverage tech-savvy customers for early adoption campaigns.")
    else:
        st.write("Adoption rate data not available for current filters.")

st.markdown("---")
st.caption("Dashboard by Sanchit Singh Thapa | MBA Research | SP Jain")
