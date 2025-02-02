<h1 align="center"> 🚀 AI-Powered Zero-Trust Access Control 🔒 </h1>

![image](https://github.com/user-attachments/assets/40f9e8c7-122d-488f-bfe5-36fc21ff50fd)

[![Logo](https://img.shields.io/badge/AI%20Powered%20Zero%2DTrust%20Access%20Control-blue?style=flat-square)](https://ai-powered-zero-trust-access-control-jkduzc6ennjgf98hdyr44y.streamlit.app/)

## ![Overview](https://img.shields.io/badge/Overview-📖-blue?style=for-the-badge)

This project implements an AI-powered Zero-Trust Access Control system that uses machine learning to assess the trust level of users based on their activity data. It predicts whether a user is "Trusted" or "Untrusted" based on device, location, activity, and risk score data. The model leverages Logistic Regression for binary classification and integrates Groq API with the **deepseek-r1-distill-llama-70b** model for advanced real-time analysis.

The app provides a web-based interface using Streamlit , allowing users to upload datasets, perform real-time predictions, visualize risk scores, and adjust access control policies dynamically.

## Features ✨
- **Data Upload** 📤: Upload your user activity data in CSV format.
- **Real-time Prediction** 🔮: Enter user activity data to predict trust level using both Logistic Regression and Groq AI Analysis.
- **Model Training** 🧠: Trains a logistic regression model to assess trust levels.
- **Groq API Integration** 🌟: Provides detailed insights into user activity using the **deepseek-r1-distill-llama-70b** model.
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

5. **Add Groq API Key** :
   - **Create a .env file in the root directory and add your Groq API key**:

   ```bash
   GROQ_API_KEY=your_api_key_here
   
6. **Run the Streamlit app**:

   ```bash
   streamlit run app.py

6. **Open the browser and navigate to http://localhost:8501 to interact with the app**.

## Usage 🚀

## Upload Data 📤
- Upload a CSV file containing user activity data with the following columns:
  - **user** : Username (e.g., user1)
  - **device** : Device name (e.g., deviceA)
  - **location** : Location (e.g., office, remote, unknown)
  - **activity** : Activity performed (e.g., login, transfer, access_data, logout)
  - **risk_score** : Risk score between 0.0 and 1.0
  - **trust_level** : Binary trust level (1 = Trusted, 0 = Untrusted)

## Real-time Trust Prediction 🔮
- **Enter values for the following parameters**:
  - **Username** (e.g., user1)
  - **Device** (e.g., deviceA)
  - **Location** (e.g., office)
  - **Activity** (e.g., login)
    
- The app will predict whether the user is Trusted or Untrusted using Logistic Regression and provide additional insights from the Groq API.

# Predicted Trust Level Trusted

![Trusted_page-0001](https://github.com/user-attachments/assets/545f575e-3bae-4e24-b316-352be72f5b32)

# Predicted Trust Level Untrusted

![Untrusted_page-0001](https://github.com/user-attachments/assets/2413d847-a16d-4e93-8bd7-f30c677d1ebd)

## Adjust Trust Threshold ⚖️
- Use the slider to adjust the trust threshold dynamically. This determines the criteria for trusting a user based on the model's risk score.

## Risk Score Distribution 📊
- The app displays a bar chart showing the distribution of risk scores across all users in the dataset.

## Dependencies ⚙️

- **Python 3.7+** 🐍
- **pandas** 🐼
- **scikit-learn** 📚
- **streamlit** 🎥
- **matplotlib** 📊
- **seaborn** 🦢
- **groq** 🌟 
- **python-dotenv**
   
## Model Details 🧠
- **Logistic Regression** :
  - The machine learning model is trained to predict the trust_level **(either Trusted or Untrusted)** based on the following features:

    - **user**
    - **device**
    - **location**
    - **activity**
    - **risk_score**

- **Groq API** :
  - The **deepseek-r1-distill-llama-70b** model is used to provide detailed, natural language-based analysis of user activity. This complements the binary prediction from Logistic Regression by offering contextual insights.

## Future Enhancements 🚀

- **Advanced Models** : Replace Logistic Regression with more sophisticated models like Random Forest or Gradient Boosting.
- **Scalability** : Optimize the app for large-scale datasets and real-time processing.
- **Security** : Implement role-based access control and secure authentication for sensitive environments.
- **Deployment** : Deploy the app on cloud platforms like AWS, Azure, or Streamlit Cloud for broader accessibility.

## 🤝 Contributing 
- We welcome contributions! If you would like to improve or add new features to this project, please fork the repository and submit a pull request.

## 🌐 Connect with Me 

- 📧 [Email](mailto:gauravghandat12@gmail.com)
- 💼 [LinkedIn](www.linkedin.com/in/gaurav-ghandat-68a5a22b4)




















