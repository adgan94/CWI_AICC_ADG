import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# StatsResortCostData_(copy).xlsx
# Read data from the Excel file
file_path = '/home/aaron/Desktop/StatsResortCostData.xlsx'
data = pd.read_excel(file_path)

x_values = data['Index']
y_values = data['% diff']

'''# Line Plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o')
plt.title('Line Plot of Price Differences', fontsize=24)
plt.xlabel('Index', fontsize=14)
plt.ylabel('Price Difference', fontsize=14)
plt.show()

# Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(x_values, y_values, color='blue')
plt.title('Bar Chart of Price Differences', fontsize=24)
plt.xlabel('Index', fontsize=14)
plt.ylabel('Price Difference', fontsize=14)
plt.show()'''

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations', fontsize=24)
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(y=y_values)
plt.title('Violin Plot of Price Differences', fontsize=24)
plt.ylabel('Price Difference', fontsize=14)
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(y_values, bins=20, color='green', edgecolor='black')
plt.title('Histogram of Price Differences', fontsize=24)
plt.xlabel('Price Difference', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# Box and Whisker Plot
plt.figure(figsize=(10, 6))
sns.boxplot(y=y_values)
plt.title('Box and Whisker Plot of Price Differences', fontsize=24)
plt.ylabel('Price Difference', fontsize=14)
plt.show()