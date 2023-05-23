from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('main page.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/res1')
def result1():
    return render_template('result1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    age = float(request.form['age'])
    gender = int(request.form['gender'])

    if age >= 15 and age <= 75:
        return render_template('result.html')
    elif age > 75:
        return render_template('result1.html')
    else:
        return render_template('main page.html')

data = pd.read_csv('cancer patient data sets.csv')
print(data)


# Drop irrelevant columns if any
data = data.drop('Patient Id', axis=1)

# Convert categorical variables to numerical using LabelEncoder
categorical_cols = ['Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards',
                    'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
                    'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                    'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
                    'Frequent Cold', 'Dry Cough', 'Snoring']
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Split into features (X) and target variable (y)
X = data.drop('Level', axis=1)
y = data['Level']

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Create the classifier
classifier = LogisticRegression()

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')



if __name__ == '__main__':
    app.run(debug=True)