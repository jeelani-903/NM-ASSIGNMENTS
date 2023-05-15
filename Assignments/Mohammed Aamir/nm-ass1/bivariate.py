import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('House Price India.csv')


# Step 2: Explore the dataset
print(df.head())
print(df.info())

# Step 3: Perform bivariate analysis

# Calculate the correlation coefficient
correlation = df['number of bedrooms'].corr(df['number of bathrooms'])
print('Correlation coefficient:', correlation)

# Create a scatter plot
plt.scatter(df['number of bedrooms'], df['number of bathrooms'])
plt.xlabel('number of bedrooms')
plt.ylabel('number of bathrooms')
plt.title('number of bedrooms vs. number of bathrooms')
plt.show()

# Perform regression analysis (optional)
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(df['number of bedrooms'], df['number of bathrooms'])
print('Regression slope:', slope)
print('Regression intercept:', intercept)
print('R-squared value:', r_value**2)