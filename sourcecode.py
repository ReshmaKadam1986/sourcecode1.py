# Required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess dataset
def load_dataset():
    # Example dataset (simplified)
    data = {
        'Technology': ['AI', 'Blockchain', 'Cloud', 'IoT', 'AI', 'Cloud', 'Blockchain', 'IoT'],
        'Risk': ['High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High'],
        'Compliance': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
    }
    df = pd.DataFrame(data)
    
    # Convert categorical variables to numerical (dummy variables)
    df = pd.get_dummies(df, columns=['Technology', 'Risk', 'Compliance'])
    
    # Split dataset into features and labels
    X = df.drop(['Compliance_No', 'Compliance_Yes'], axis=1)  # Features
    y = df['Compliance_Yes']  # Labels
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    
    # Train the model
    clf.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Main function to run the script
if __name__ == "__main__":
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset()
    
    # Train and evaluate model
    train_and_evaluate(X_train, X_test, y_train, y_test)

Output:
[Running] python -u "c:\Users\User\Documents\Reshma\sourcecode.py"
Accuracy: 1.00
Classification Report:
       precision    recall  f1-score   support

       False       1.00      1.00      1.00         1
        True       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2
