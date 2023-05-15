import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("penguins_size.csv")

variable_of_interest = 'species'

# Step 5: Summarize categorical variables
category_counts = df[variable_of_interest].value_counts()
print(category_counts)

# Plot a bar chart
category_counts.plot(kind='bar')
plt.xlabel(variable_of_interest)
plt.ylabel('Count')
plt.title('Distribution of ' + variable_of_interest)
plt.show()

# Step 2: Explore the dataset
print(df.head())
print(df.info())

# Step 3: Perform bivariate analysis

# Calculate the correlation coefficient
correlation = df['culmen_depth_mm'].corr(df['flipper_length_mm'])
print('Correlation coefficient:', correlation)

# Create a scatter plot
plt.scatter(df['culmen_depth_mm'], df['flipper_length_mm'])
plt.xlabel('culmen_depth_mm')
plt.ylabel('flipper_length_mm')
plt.title('culmen_depth_mm vs. flipper_length_mm')
plt.show()

# Perform regression analysis (optional)
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(df['species'], df['island'])
print('Regression slope:', slope)
print('Regression intercept:', intercept)
print('R-squared value:', r_value**2)

# Step 2: Explore the dataset
print(df.head())
print(df.info())

# Step 3: Perform multivariate analysis

# Create a scatter plot matrix
plt.scatter(df['culmen_length_mm'], df['flipper_length_mm'], c=df['body_mass_g'], cmap='viridis')
plt.colorbar(label='z')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Multivariate Scatter Plot')
plt.show()

variable_of_interest = "body_mass_g"

# Step 2: Generate descriptive statistics
descriptive_stats = df.describe()

# Step 3: Print the descriptive statistics
print(descriptive_stats)

# For numerical variables
descriptive_stats = df[variable_of_interest].describe()
print(descriptive_stats)

# Generate a histogram
df[variable_of_interest].hist()
plt.xlabel(variable_of_interest)
plt.ylabel('Frequency')
plt.title('Distribution of ' + variable_of_interest)
plt.show()

# Generate a box plot
df[variable_of_interest].plot(kind='box')
plt.ylabel(variable_of_interest)
plt.title('Box Plot of ' + variable_of_interest)
plt.show()

# Calculate the mean and standard deviation of the dataset
mean = np.mean(df)
std = np.std(df)

# Define a threshold to determine outliers (e.g., 3 standard deviations away from the mean)
threshold = 3

# Find the outliers by comparing each value with the threshold
outliers = df[(np.abs(df - mean) > threshold * std).astype(bool)]

# Replace the outliers with a specific value (e.g., the mean of the dataset)
df = df.mask(np.abs(df - mean) > threshold * std, mean, axis=0)

# Print the dataset with outliers replaced
print("Dataset with outliers replaced:")
print(df)

# Select the columns to be scaled (numeric columns)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Perform min-max scaling on the selected columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Print the scaled dataset
print("Scaled dataset:")
print(df)

# Step 4: Select the features (independent variables) for clustering
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']  # Replace with your desired features/columns

# Step 5: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Step 6: Perform the DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust the hyperparameters as needed
clusters = dbscan.fit_predict(scaled_data)

# Step 7: Add the cluster data with the primary dataset
data_with_clusters = df.copy()
data_with_clusters['Cluster'] = clusters

# Step 8: Visualize the clusters (example: using scatter plot for two features)
plt.scatter(data_with_clusters['culmen_length_mm'], data_with_clusters['culmen_depth_mm'], c=data_with_clusters['Cluster'], cmap='viridis')
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_depth_mm')
plt.title('DBSCAN Clustering')
plt.show()

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Perform label encoding on categorical columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Print the updated dataset with encoded categorical columns
print("Updated dataset with encoded categorical columns:")
print(df)

# Step 3: Split the data into dependent and independent variables
X = df.drop('body_mass_g', axis=1)  # Independent variables
y = df['body_mass_g']  # Dependent variable

# Step 4: Print the shapes of the resulting datasets
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Step 3: Split the data into dependent and independent variables
# Assuming the target variable column is named 'TargetVariable'
X = df.drop('body_mass_g', axis=1)  # Independent variables
y = df['body_mass_g']  # Dependent variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the model (Logistic Regression in this example)
model = LogisticRegression()

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Test the model
predictions = model.predict(X_test)

# Step 8: Measure the performance using evaluation metrics (example: accuracy score)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Step 3: Split the data into dependent and independent variables
X = df.drop('body_mass_g', axis=1)
y = df['body_mass_g']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the model (Logistic Regression in this example)
model = LogisticRegression()

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Test the model
predictions = model.predict(X_test)

# Step 8: Evaluate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

# Step 9: Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC AUC Score:", roc_auc)
