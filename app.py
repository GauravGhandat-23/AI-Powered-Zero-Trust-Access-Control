import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Data (Replace with actual data)
data = {
    'user': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'device': ['deviceA', 'deviceB', 'deviceC', 'deviceA', 'deviceB'],
    'location': ['office', 'remote', 'remote', 'office', 'remote'],
    'activity': ['login', 'transfer', 'access_data', 'login', 'access_data'],
    'risk_score': [0.2, 0.8, 0.1, 0.5, 0.9],
    'trust_level': [1, 0, 1, 1, 0]
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

# Streamlit App
def main():
    st.title("AI-Powered Zero-Trust Access Control")
    st.image("logo.jpeg", width=500)

    # Data Upload
    uploaded_file = st.file_uploader("Upload user activity data (CSV)")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Handle case where no file is uploaded (optional: show a message)
        return

    # Preprocess Data
    df_processed = preprocess_data(df)

    # Train Model
    model, train_columns, accuracy = train_model(df_processed)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # User Input for Real-time Assessment
    user_input = st.text_input("Enter username:")
    device_input = st.text_input("Enter device name:")
    location_input = st.selectbox("Select location:", ['office', 'remote'])
    activity_input = st.selectbox("Select activity:", ['login', 'transfer', 'access_data', 'logout'])

    # Predict Trust Level
    if user_input and device_input and location_input and activity_input:
        # Ensure new_data is always initialized
        new_data = {
            'user': [user_input],
            'device': [device_input],
            'location': [location_input],
            'activity': [activity_input],
            'risk_score': [0.5]  # You can assign a default value to risk_score
        }
        new_df = pd.DataFrame(new_data)
        new_df_processed = preprocess_data(new_df)

        # Align the columns of new_df_processed with the model's training data
        missing_cols = set(train_columns) - set(new_df_processed.columns)
        for col in missing_cols:
            new_df_processed[col] = 0  # Add missing columns with default value of 0
        new_df_processed = new_df_processed[train_columns]  # Ensure column order matches

        # Predict trust level
        trust_prediction = model.predict(new_df_processed)[0]
        st.write(f"Predicted Trust Level: {'Trusted' if trust_prediction == 1 else 'Untrusted'}")

    # Visualization (Example: Risk Score Distribution)
    st.header("Risk Score Distribution")
    st.bar_chart(df['risk_score'])

    # Policy Adjustments (Example: Threshold for Trust)
    st.header("Access Control Policies")
    trust_threshold = st.slider("Trust Threshold:", 0.0, 1.0, 0.5, 0.1)
    st.write(f"Current Trust Threshold: {trust_threshold}")

if __name__ == "__main__":
    main()
