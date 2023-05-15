import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('House Price India.csv')

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
mean_value = df['Distance from the airport'].mean()
dataset_filled_mean = df['Distance from the airport'].fillna(mean_value)

median_value = df['Distance from the airport'].median()
dataset_filled_median = df['Distance from the airport'].fillna(median_value)

mode_value = df['Distance from the airport'].mode()[0]
dataset_filled_mode = df['Distance from the airport'].fillna(mode_value)

# Remember to replace 'column_name' with the actual name of the column containing missing values.

# Step 4: Verify if missing values have been handled
print(dataset_filled.isnull().sum())