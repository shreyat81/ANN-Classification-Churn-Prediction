**Customer Churn Prediction Using Artificial Neural Networks**

This project predicts customer churn using an Artificial Neural Network (ANN) trained on the popular Churn_Modelling.csv dataset. The goal is to determine whether a customer is likely to leave a bank, based on features like geography, balance, credit score, etc.

**Project Structure:**
annclassification/
├── Churn_Modelling.csv               # Dataset
├── prediction.ipynb                  # Inference using trained model
├── hyperparametertuningann.ipynb    # Hyperparameter tuning and ANN training
├── experiments.ipynb                # Model evaluation and experiment logs
├── model.h5                          # Trained ANN model
├── scaler.pkl                        # StandardScaler used on features
├── label_encoder_gender.pkl         # Encoder for 'Gender' column
├── onehot_encoder_geo.pkl           # Encoder for 'Geography' column
├── salaryregression.ipynb           # Additional regression experiment (optional)
├── requirements.txt                 # Dependencies
└── README.md                        # Project Documentation

**🔍 Problem Statement**

Churn is the percentage of service subscribers who discontinue their subscriptions within a given time period. Reducing churn is critical for business growth. This project uses deep learning (ANN) to classify whether a customer will churn based on features like:
	•	Credit Score
	•	Age
	•	Balance
	•	Gender
	•	Geography
	•	Tenure
	•	Estimated Salary
	•	Number of Products
	•	Active Membership

⸻

**📊 Dataset**
	•	Source: Churn_Modelling.csv
	•	Samples: 10,000 customer records
	•	Target: Exited column (0 = not churned, 1 = churned)

⸻

**🔧 Preprocessing**
	•	Feature Engineering:
	•	Label Encoding for Gender
	•	OneHot Encoding for Geography
	•	Scaling:
	•	Features scaled using StandardScaler
	•	Saved Models:
	•	Encoders and scalers are saved using pickle for reuse during inference.

⸻

**🧪 Model Development**
	•	Frameworks Used: TensorFlow + Keras (via scikeras)
	•	Model Architecture:
	•	Input layer based on encoded features
	•	Two hidden layers (with ReLU activation)
	•	Output layer with sigmoid activation
	•	Loss Function: Binary Crossentropy
	•	Optimizer: Adam
	•	Evaluation Metrics: Accuracy, Precision, Recall, F1-score, AUC

⸻

**🔁 Hyperparameter Tuning**

Performed using GridSearchCV with KerasClassifier via scikeras. Tuned parameters:
	•	Number of neurons
	•	Batch size
	•	Epochs
	•	Optimizer choice

Results and best parameters are documented in hyperparametertuningann.ipynb.

⸻

**🧪 Experiments & Results**

Model evaluation includes:
	•	Confusion Matrix
	•	ROC Curve
	•	Classification Report
	•	Accuracy trends across different hyperparameter combinations

Refer to experiments.ipynb for detailed analysis.

⸻

**🚀 Inference**

To predict churn for new customer data:
	1.	Apply the same label/one-hot encoding.
	2.	Scale input using scaler.pkl
	3.	Load model.h5 and make predictions using prediction.ipynb.

⸻

**💻 Tech Stack**
	•	Python 3.x
	•	TensorFlow 2.15.0
	•	Scikit-learn
	•	Pandas, NumPy, Matplotlib
	•	Scikeras
	•	Streamlit (optional for web deployment)

**🧪 Running the Notebooks**

Open and execute:
	•	hyperparametertuningann.ipynb: To tune/train the model
	•	experiments.ipynb: To review model performance
	•	prediction.ipynb: To make predictions on new data

⸻

🌐 Future Improvements
	•	Add Streamlit-based web UI for interactive prediction
	•	Deploy as Flask/FastAPI backend service
	•	Try ensemble models (XGBoost, LightGBM) for comparison
