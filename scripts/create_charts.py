import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

# Parameters for the plot
metrics_names = ['accuracy_top1', 'accuracy_top3', 'accuracy_top5']
title = 'Comparison of Retrieval Methods on Various Metrics with Advanced Embedding Model'

# Experiment values and labels (manually ordered)
experiment_values = [
    [0.4, 0.64, 0.67],  # Hybrid Search, with routing, with Reranker (Blue)
    [0.4, 0.64, 0.67],  # Hybrid Search, no routing (Purple)
    [0.52, 0.62, 0.65],  # No Hybrid Search, with Reranker (Gray)
    [0.48, 0.61, 0.64],  # No Hybrid Search, without Reranker (Orange)
]

legend_labels = [
    'Hybrid Search, with routing, with Reranker',
    'Hybrid Search, no routing',
    'No Hybrid Search, with Reranker',
    'No Hybrid Search, without Reranker'
]

# Colors for each experiment (customize as needed)
colors = [
    (np.array([0.27, 0.53, 1.0]), np.array([0.0, 0.25, 0.75])),  # Gradient for Hybrid Search, with routing, with Reranker
    (np.array([0.5, 0.0, 0.5]), np.array([0.3, 0.0, 0.3])),  # Gradient for Hybrid Search, no routing
    (np.array([0.3, 0.3, 0.3]), None),  # Solid color for No Hybrid Search, with Reranker
    (np.array([0.9, 0.6, 0.1]), np.array([0.7, 0.4, 0.1])),  # Gradient for No Hybrid Search, without Reranker
]


x = np.arange(len(metrics_names))  # the label locations
width = 0.25 / len(experiment_values)  # the width of the bars, adjusted for the number of experiments

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0f111a')
fig.patch.set_facecolor('#0f111a')

# Function to create gradients
def create_gradient(ax, x, y, width, color1, color2):
    for xi, yi in zip(x, y):
        if np.isnan(yi):
            continue
        gradient = np.linspace(0, 1, 256)
        for i, t in enumerate(gradient):
            rect = patches.Rectangle(
                (xi - width / 2, yi * i / 256), width, yi / 256,
                color=color1 * (1 - t) + color2 * t, transform=ax.transData,
                zorder=1)
            ax.add_patch(rect)
            
# Add labels above the bars
for i, values in enumerate(experiment_values):
    for j, value in enumerate(values):
        ax.text(x[j] + i * width - width * len(experiment_values) / 2, value + 0.01, f'{value:.2f}', ha='center', color='white', fontsize=8)

# Plot each experiment
for i, (values, color) in enumerate(zip(experiment_values, colors)):
    if color[1] is not None:  # Gradient
        create_gradient(ax, x + i * width - width * len(experiment_values) / 2, values, width, *color)
    else:  # Solid color
        ax.bar(x + i * width - width * len(experiment_values) / 2, values, width, color=color[0], edgecolor='none', zorder=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score', color='white')
ax.set_title(title, color='white')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, color='white')

# Legend
legend_colors = [color[0] if color[1] is None else (color[0] + color[1]) / 2 for color in colors]  # Compute legend color
handles = [patches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
ax.legend(handles=handles, facecolor='#0f111a', edgecolor='none', loc='upper center', frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.15), ncol=len(legend_labels), labelcolor='white')

# Set the background color of the axes
ax.set_facecolor('#0f111a')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='x', colors='white')

# Customize gridlines
ax.yaxis.grid(True, color='#2a2a2a', linestyle='--')
ax.set_axisbelow(True)
ax.xaxis.grid(False)
ax.spines['bottom'].set_color('#2a2a2a')
ax.spines['top'].set_color('#0f111a')
ax.spines['right'].set_color('#0f111a')
ax.spines['left'].set_color('#2a2a2a')

fig.tight_layout()

# Display the plot
plt.savefig('create_charts_result.png', facecolor='#0f111a', bbox_inches='tight')
plt.show()