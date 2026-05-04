import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read data from the Excel file
file_path = '/home/aaron/Desktop/StatsResortCostData.xlsx'
data = pd.read_excel(file_path)

# Assuming the Excel file has columns 'A' and 'G'
x_values = data['Index']
y_values = data['% diff']

# Create a mask for zero y values
zero_mask = y_values == 0

# Create a color array, setting red for zero y values and using a colormap for others
colors = np.where(zero_mask, 'red', 'blue')

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots()
scatter = ax.scatter(x_values, y_values, c=colors, s=50)

# Set chart title and label axes
ax.set_title("Holiday vs Non-holiday Price Diff.", fontsize=24)
ax.set_xlabel("Index", fontsize=14)
ax.set_ylabel("Price Diff.", fontsize=14)

# Set size of tick labels
ax.tick_params(labelsize=14)

# Set range for each axis
ax.axis([0, 45, -1, 20])

# Hide the axes if needed
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)

plt.show()

