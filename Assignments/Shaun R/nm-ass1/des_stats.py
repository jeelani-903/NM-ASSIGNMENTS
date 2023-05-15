import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('House Price India.csv')

variable_of_interest = "Distance from the airport"

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



