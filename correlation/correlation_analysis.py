import numpy as np
import matplotlib.pyplot as plt

# 1. Data provided in the assignment
x = np.array([
    -10.0, -8.5, -6.20, -4.3, -2.0, 0.5, 2.0, 4.0, 6.0, 8.0, 10.0
])

y = np.array([
    -7.0, -4.0, -5.0, -1.0, -0.5, 1.0, 2.0, 3.0, 3.0, 5.0, 6.0
])

# 2. Calculate Pearson's correlation coefficient
# np.corrcoef returns a matrix; [0,1] is the value between x and y
correlation_matrix = np.corrcoef(x, y)
pearson_r = correlation_matrix[0, 1]

print(f"--- Task 1 Results ---")
print(f"Pearson's Correlation Coefficient: {pearson_r:.4f}")

# 3. Create the visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points (Blue Dots)')

# Add a regression line (line of best fit)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Trend Line (r={pearson_r:.2f})')

# Formatting the graph
plt.title('Visualization of Data Points and Correlation', fontsize=14)
plt.xlabel('X Values', fontsize=12)
plt.ylabel('Y Values', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 4. Save the plot for your GitHub report
plt.savefig('correlation_plot.png')
print("Graph saved as 'correlation_plot.png'")

plt.show()
