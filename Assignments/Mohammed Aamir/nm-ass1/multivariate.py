import pandas as pd
import matplotlib.pyplot as plt
# Step 1: Load the dataset
df = pd.read_csv('House Price India.csv')

# Step 2: Explore the dataset
print(df.head())
print(df.info())

# Step 3: Perform multivariate analysis

# Create a scatter plot matrix
plt.scatter(df['Area of the house(excluding basement)'], df['Area of the basement'], c=df['Number of schools nearby'], cmap='viridis')
plt.colorbar(label='z')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Multivariate Scatter Plot')
plt.show()