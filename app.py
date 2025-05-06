import datetime
import logging
import os
from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Startup environment/config validation ---
REQUIRED_ENV_VARS = ["JWT_SECRET_KEY", "ENCRYPTION_KEY", "OPENAI_API_KEY"]
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if ENVIRONMENT == "production" and missing:
    raise RuntimeError(
        f"Missing required environment variables in production: {', '.join(missing)}"
    )
if ENVIRONMENT != "production" and missing:
    logging.warning(
        f"[DEV] Missing secrets: {', '.join(missing)}. Set these before deploying to production!"
    )

# CORS and Rate Limit enforcement
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY", "")

# App title and configuration
st.set_page_config(
    page_title="CustomerAI Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Authentication
def authenticate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.user_role = None
        st.session_state.token = None

    if not st.session_state.authenticated:
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            try:
                response = requests.post(
                    f"{API_URL}/auth/login",
                    json={"username": username, "password": password},
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.authenticated = True
                    st.session_state.token = data["data"]["token"]
                    st.session_state.user_id = data["data"]["user_id"]
                    st.session_state.user_role = data["data"]["roles"]
                    st.sidebar.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Invalid credentials")
            except Exception as e:
                st.sidebar.error(f"Connection error: {str(e)}")

        # Demo mode
        if st.sidebar.button("Demo Mode"):
            st.session_state.authenticated = True
            st.session_state.token = "demo_token"
            st.session_state.user_id = "demo_user"
            st.session_state.user_role = ["demo"]
            st.experimental_rerun()

        return False
    return True


# API request helper
def api_request(endpoint, method="GET", data=None, params=None):
    headers = (
        {"Authorization": f"Bearer {st.session_state.token}"}
        if st.session_state.token != "demo_token"
        else {}
    )

    if st.session_state.token == "demo_token":
        # Return demo data
        if endpoint == "analytics/summary":
            return {
                "data": {
                    "total_conversations": 12542,
                    "sentiment_distribution": {
                        "positive": 0.65,
                        "negative": 0.18,
                        "neutral": 0.17,
                    },
                    "average_satisfaction": 4.2,
                    "response_time_avg": 3.7,
                    "resolution_rate": 0.92,
                    "timeline_data": [
                        {
                            "date": (datetime.datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                            "conversations": int(400 - i * 10 + np.random.randint(-30, 30)),
                            "avg_sentiment": 0.6 + np.random.random() * 0.2,
                        }
                        for i in range(14)
                    ],
                    "top_issues": [
                        {"issue": "account_access", "count": 342, "sentiment": 0.42},
                        {
                            "issue": "transaction_disputes",
                            "count": 289,
                            "sentiment": 0.37,
                        },
                        {"issue": "loan_inquiries", "count": 187, "sentiment": 0.67},
                        {"issue": "mobile_app", "count": 156, "sentiment": 0.51},
                        {"issue": "fee_questions", "count": 134, "sentiment": 0.39},
                    ],
                }
            }
        elif endpoint == "review/queue":
            return {
                "data": {
                    "items": [
                        {
                            "item_id": f"rev-{i}",
                            "query": "What stocks should I invest in for maximum returns?",
                            "response": "Based on current market trends, diversification is recommended...", # Shortened for brevity
                            "category": "investment_advice",
                            "priority": 2,
                            "timestamp": (datetime.datetime.now() - timedelta(hours=i)).isoformat(),
                        }
                        for i in range(1, 6)
                    ]
                }
            }

    try:
        url = f"{API_URL}/{endpoint}"
        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


# Sidebar navigation
def sidebar():
    st.sidebar.title("CustomerAI Insights")

    if st.session_state.user_role:
        st.sidebar.write(f"User: {st.session_state.user_id}")
        st.sidebar.write(f"Role: {', '.join(st.session_state.user_role)}")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Sentiment Analysis",
            "Response Generation",
            "Human Review Queue",
            "Fairness Analysis",
            "Privacy Tools",
        ],
    )

    if st.sidebar.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

    return page


# Dashboard page
def dashboard_page():
    st.title("CustomerAI Insights Dashboard")

    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.DateProvider.get_instance().now() - timedelta(days=14),
        )
    with col2:
        end_date = st.date_input("End Date", datetime.DateProvider.get_instance().now())

    # Get analytics data
    response = api_request(
        "analytics/summary",
        params={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
    )

    if response:
        data = response["data"]

        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Conversations", f"{data['total_conversations']:,}")
        col2.metric("Avg. Satisfaction", f"{data['average_satisfaction']:.1f}/5.0")
        col3.metric("Avg. Response Time", f"{data['response_time_avg']:.1f}h")
        col4.metric("Resolution Rate", f"{data['resolution_rate']*100:.1f}%")

        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_data = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Percentage": [
                    data["sentiment_distribution"]["positive"] * 100,
                    data["sentiment_distribution"]["negative"] * 100,
                    data["sentiment_distribution"]["neutral"] * 100,
                ],
            }
        )

        st.bar_chart(sentiment_data.set_index("Sentiment"))

        # Timeline chart
        st.subheader("Conversation Volume & Sentiment Trend")
        timeline_data = pd.DataFrame(data["timeline_data"])
        timeline_data["date"] = pd.to_datetime(timeline_data["date"])

        base = alt.Chart(timeline_data).encode(x="date:T")
        line = base.mark_line(color="blue").encode(y="avg_sentiment:Q")
        bars = base.mark_bar(color="lightblue").encode(y="conversations:Q")

        st.altair_chart(
            alt.layer(bars, line).resolve_scale(y="independent"),
            use_container_width=True,
        )

        # Top issues
        st.subheader("Top Customer Issues")
        issues_data = pd.DataFrame(data["top_issues"])

        fig = (
            alt.Chart(issues_data)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Number of Conversations"),
                y=alt.Y("issue:N", sort="-x", title="Issue Category"),
                color=alt.Color(
                    "sentiment:Q",
                    scale=alt.Scale(scheme="redblue"),
                    title="Sentiment Score",
                ),
            )
            .properties(height=200)
        )

        st.altair_chart(fig, use_container_width=True)


# Sentiment Analysis page
def sentiment_page():
    st.title("Sentiment Analysis")

    text = st.text_area("Enter customer text for analysis", height=150)

    if st.button("Analyze Sentiment"):
        if text:
            with st.spinner("Analyzing..."):
                response = api_request("analyze/sentiment", method="POST", data={"text": text})

                if response:
                    data = response["data"]

                    col1, col2, col3 = st.columns(3)
                    sentiment = data["sentiment"].capitalize()
                    col1.metric("Overall Sentiment", sentiment)
                    col2.metric(
                        "Satisfaction Score",
                        f"{data['analysis']['satisfaction_score']}/10",
                    )

                    # Sentiment scores
                    st.subheader("Sentiment Breakdown")
                    scores_data = pd.DataFrame(
                        {
                            "Sentiment": ["Positive", "Negative", "Neutral"],
                            "Score": [
                                data["positive"],
                                data["negative"],
                                data["neutral"],
                            ],
                        }
                    )

                    st.bar_chart(scores_data.set_index("Sentiment"))

                    # Key positives/negatives
                    if "key_positives" in data["analysis"]:
                        st.subheader("Key Positive Aspects")
                        for item in data["analysis"]["key_positives"]:
                            st.success(item)

                    if "key_negatives" in data["analysis"]:
                        st.subheader("Areas for Improvement")
                        for item in data["analysis"].get("key_negatives", []):
                            st.error(item)
        else:
            st.warning("Please enter some text to analyze")


# Main application
def main():
    if authenticate():
        page = sidebar()

        if page == "Dashboard":
            dashboard_page()
        elif page == "Sentiment Analysis":
            sentiment_page()
        elif page == "Response Generation":
            st.title("Response Generation")
            st.info("This section allows generating compliant responses for customer queries")
            # Response generation implementation...
        elif page == "Human Review Queue":
            st.title("Human Review Queue")
            st.info("Review queue for high-risk responses that require human approval")
            # Human review implementation...
        elif page == "Fairness Analysis":
            st.title("Fairness Analysis")
            st.info("Tools to detect and mitigate bias across demographic groups")
            # Fairness analysis implementation...
        elif page == "Privacy Tools":
            st.title("Privacy Protection")
            st.info("Tools for PII detection and anonymization")
            # Privacy tools implementation...


if __name__ == "__main__":
    main()
