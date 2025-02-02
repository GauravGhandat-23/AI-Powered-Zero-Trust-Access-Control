import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
from groq import Groq

# Load Dataset
data = {
    'user': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10',
             'user11', 'user12', 'user13', 'user14', 'user15', 'user16', 'user17', 'user18', 'user19', 'user20'],
    'device': ['deviceA', 'deviceB', 'deviceC', 'deviceA', 'deviceB', 'deviceD', 'deviceE', 'deviceF', 'deviceG', 'deviceH',
               'deviceA', 'deviceB', 'deviceC', 'deviceD', 'deviceE', 'deviceF', 'deviceG', 'deviceH', 'deviceA', 'deviceB'],
    'location': ['office', 'remote', 'remote', 'office', 'remote', 'unknown', 'office', 'remote', 'office', 'unknown',
                 'office', 'remote', 'office', 'remote', 'office', 'remote', 'office', 'unknown', 'remote', 'office'],
    'activity': ['login', 'transfer', 'access_data', 'login', 'access_data', 'logout', 'login', 'transfer', 'access_data', 'logout',
                 'transfer', 'login', 'logout', 'access_data', 'login', 'logout', 'transfer', 'access_data', 'login', 'logout'],
    'risk_score': [0.2, 0.8, 0.1, 0.5, 0.9, 0.4, 0.3, 0.7, 0.2, 0.6,
                   0.5, 0.8, 0.3, 0.9, 0.1, 0.6, 0.4, 0.7, 0.2, 0.5],
    'trust_level': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Feature Engineering
def preprocess_data(df):
    df = pd.get_dummies(df, columns=['user', 'device', 'location', 'activity'], dummy_na=True)
    return df

# Train a Machine Learning Model (Logistic Regression as an example)
def train_model(df):
    X = df.drop('trust_level', axis=1)
    y = df['trust_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, X_train.columns, accuracy

# Load environment variables from .env file
load_dotenv()

def query_groq_api(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the .env file.")
    
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response

# Streamlit App
def main():
    st.title("AI-Powered Zero-Trust Access Control")
    st.image("logo.jpeg", width=500)

    # Preprocess Data
    df_processed = preprocess_data(df)

    # Train Model
    model, train_columns, accuracy = train_model(df_processed)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # User Input for Real-time Assessment
    st.header("Real-Time Trust Level Prediction")
    user_input = st.text_input("Enter username:")
    device_input = st.text_input("Enter device name:")
    location_input = st.selectbox("Select location:", ['office', 'remote', 'unknown'])
    activity_input = st.selectbox("Select activity:", ['login', 'transfer', 'access_data', 'logout'])
    risk_score_input = st.number_input("Enter risk score (0.0 to 1.0):", min_value=0.0, max_value=1.0, value=0.5)

    if user_input and device_input and location_input and activity_input:
        # Prepare new input data
        new_data = {
            'user': [user_input],
            'device': [device_input],
            'location': [location_input],
            'activity': [activity_input],
            'risk_score': [risk_score_input]
        }
        new_df = pd.DataFrame(new_data)
        new_df_processed = preprocess_data(new_df)

        # Align columns with training data
        missing_cols = set(train_columns) - set(new_df_processed.columns)
        for col in missing_cols:
            new_df_processed[col] = 0  # Add missing columns with default value of 0
        new_df_processed = new_df_processed[train_columns]  # Ensure column order matches

        # Predict trust level using logistic regression
        trust_prediction = model.predict(new_df_processed)[0]
        st.write(f"Predicted Trust Level (Logistic Regression): {'Trusted' if trust_prediction == 1 else 'Untrusted'}")

        # Query Groq API for additional insights
        prompt = (
            f"Analyze the following user activity data:\n"
            f"Username: {user_input}\n"
            f"Device: {device_input}\n"
            f"Location: {location_input}\n"
            f"Activity: {activity_input}\n"
            f"Risk Score: {risk_score_input}\n"
            f"Provide a detailed assessment of whether this user should be trusted."
        )
        groq_response = query_groq_api(prompt)
        st.subheader("Groq AI Analysis")
        st.write(groq_response)

    # Visualization (Example: Risk Score Distribution)
    st.header("Risk Score Distribution")
    st.bar_chart(df['risk_score'])

    # Policy Adjustments (Example: Threshold for Trust)
    st.header("Access Control Policies")
    trust_threshold = st.slider("Trust Threshold:", 0.0, 1.0, 0.5, 0.1)
    st.write(f"Current Trust Threshold: {trust_threshold}")

if __name__ == "__main__":
    main()
