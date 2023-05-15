import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Housing.csv")


# Step 4: Select a variable of interest
variable_of_interest = 'price'

# Step 5: Summarize categorical variables
category_counts = df[variable_of_interest].value_counts()
print(category_counts)

# Plot a bar chart
category_counts.plot(kind='bar')
plt.xlabel(variable_of_interest)
plt.ylabel('Count')
plt.title('Distribution of ' + variable_of_interest)
plt.show()
# Step 1: Load the dataset
df = pd.read_csv('Housing.csv')


# Step 2: Explore the dataset
print(df.head())
print(df.info())

# Step 3: Perform bivariate analysis

# Calculate the correlation coefficient
correlation = df['area'].corr(df['parking'])
print('Correlation coefficient:', correlation)

# Create a scatter plot
plt.scatter(df['area'], df['parking'])
plt.xlabel('area')
plt.ylabel('parking')
plt.title('area vs. parking')
plt.show()

# Perform regression analysis (optional)
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(df['area'], df['parking'])
print('Regression slope:', slope)
print('Regression intercept:', intercept)
print('R-squared value:', r_value**2)

# Step 2: Explore the dataset
print(df.head())
print(df.info())

# Step 3: Perform multivariate analysis

# Create a scatter plot matrix
plt.scatter(df['area'], df['bedrooms'], c=df['bathrooms'], cmap='viridis')
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

# Step 2: Check for missing values
print(df.isnull().sum())

# Step 3: Handle missing values
# Option 1: Drop rows with missing values
dataset_dropped = df.dropna()
print(dataset_dropped.shape)

# Option 2: Fill missing values with a specific value
dataset_filled = df.fillna(0)  # Replace missing values with 0
print(dataset_filled.isnull().sum())

# Option 3: Fill missing values with mean, median, or mode
mean_value = df['area'].mean()
dataset_filled_mean = df['area'].fillna(mean_value)

median_value = df['area'].median()
dataset_filled_median = df['area'].fillna(median_value)

mode_value = df['area'].mode()[0]
dataset_filled_mode = df['area'].fillna(mode_value)

# Remember to replace 'column_name' with the actual name of the column containing missing values.

# Step 4: Verify if missing values have been handled
print(dataset_filled.isnull().sum())

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
X = data.drop('area', axis=1)  # Independent variables
y = data['area']  # Dependent variable

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

X = data.drop('area', axis=1)
y = data['area']

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
