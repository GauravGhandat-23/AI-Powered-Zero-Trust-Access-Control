<h1 align="center"> 🚀 AI-Powered Zero-Trust Access Control 🔒 </h1>

![image](https://github.com/user-attachments/assets/40f9e8c7-122d-488f-bfe5-36fc21ff50fd)

[![Logo](https://img.shields.io/badge/AI%20Powered%20Zero%2DTrust%20Access%20Control-blue?style=flat-square)](https://ai-powered-zero-trust-access-control-jkduzc6ennjgf98hdyr44y.streamlit.app/)

## ![Overview](https://img.shields.io/badge/Overview-📖-blue?style=for-the-badge)

This project implements an AI-powered Zero-Trust Access Control system that uses machine learning to assess the trust level of users based on their activity data. It predicts whether a user is "Trusted" or "Untrusted" based on device, location, activity, and risk score data. The model leverages Logistic Regression and can be used in real-time scenarios via a web-based Streamlit application.

## Features ✨
- **Data Upload** 📤: Upload your user activity data in CSV format.
- **Real-time Prediction** 🔮: Enter user activity data to predict trust level.
- **Model Training** 🧠: Trains a logistic regression model to assess trust.
- **Visualizations** 📊: Displays risk score distribution and access control policies.
- **Policy Adjustment** ⚖️: Allows configuration of the trust threshold for decisions.

## Installation 🛠️

To run the project locally, follow the steps below:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/AI-Powered-Zero-Trust-Access-Control.git
   cd AI-Powered-Zero-Trust-Access-Control

2. **Set up the virtual environment**:

   ```bash
   python -m venv venv

3. **Activate the virtual environment**:
   - **On Windows**:
     ```bash
     venv\Scripts\activate
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate

4. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt

5. **Run the Streamlit app**:

   ```bash
   streamlit run app.py

6. **Open the browser and navigate to http://localhost:8501 to interact with the app**.

## Usage 🚀

## Upload Data 📤
- Upload a CSV file containing user activity data with the columns **user, device, location, activity, risk_score, and trust_level**.

## Real-time Trust Prediction 🔮
- **Enter values for the following parameters**:
  - **Username** (e.g., user1)
  - **Device** (e.g., deviceA)
  - **Location** (e.g., office)
  - **Activity** (e.g., login)
    
- The model will predict whether the user is "Trusted" or "Untrusted" based on these inputs.

# Predicted Trust Level Trusted

![Predicted Trust Level Trusted_page-0001](https://github.com/user-attachments/assets/a776b925-8324-4531-a869-2e261c06add3)

# Predicted Trust Level Untrusted

![Predicted Trust Level Untrusted_page-0001](https://github.com/user-attachments/assets/1d8fceee-5ea7-4286-b655-a93cb36c3230)

## Adjust Trust Threshold ⚖️
- Adjust the trust threshold slider to change the criteria for trusting a user based on the model's risk score.

## Risk Score Distribution 📊
- The app will display a bar chart of the risk scores for all users in the dataset.

## Dependencies ⚙️

- **Python 3.7+** 🐍
- **pandas** 🐼
- **scikit-learn** 📚
- **streamlit** 🎥
- **matplotlib** 📊
- **seaborn** 🦢

## Model Details 🧠
- The machine learning model is based on Logistic Regression and is trained to predict the trust_level (either trusted or untrusted) based on the following features:

  - **user**
  - **device**
  - **location**
  - **activity**
  - **risk_score**

- The model is trained and tested using the dataset, and predictions are made by entering real-time user data via the Streamlit interface.

## 🤝 Contributing 
- We welcome contributions! If you would like to improve or add new features to this project, please fork the repository and submit a pull request.

## 🌐 Connect with Me 

- 📧 [Email](mailto:gauravghandat12@gmail.com)
- 💼 [LinkedIn](www.linkedin.com/in/gaurav-ghandat-68a5a22b4)




















