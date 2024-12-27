import matplotlib.pyplot as plt
from math import cos, sin, pi  # Import trigonometric functions from math

# Fabricated data
labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [25, 35, 20, 20]  # Percentages for each category
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']  # Colors for the pie chart
explode = (0.05, 0.05, 0.05, 0.05)  # Slightly "explode" all parts for clarity

# Create the pie chart
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.8,
    wedgeprops={'edgecolor': 'black'}
)

# Add lines connecting percentages to their sections
for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2  # Calculate angle of the section
    x = wedge.r * 0.8 * cos(angle * pi / 180)  # X-coordinate for line
    y = wedge.r * 0.8 * sin(angle * pi / 180)  # Y-coordinate for line
    ax.plot([0, x], [0, y], color='black', lw=1)  # Draw line to the label

# Beautify the chart
plt.title('Pie Chart with Line Indicators for Percentages')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular

# Show the plot
plt.show()
