import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("House Price India.csv")

# Step 4: Select a variable of interest
variable_of_interest = 'Number of schools nearby'

# Step 5: Summarize categorical variables
category_counts = df[variable_of_interest].value_counts()
print(category_counts)

# Plot a bar chart
category_counts.plot(kind='bar')
plt.xlabel(variable_of_interest)
plt.ylabel('Count')
plt.title('Distribution of ' + variable_of_interest)
plt.show()

# Step 6: Summarize numerical variables
