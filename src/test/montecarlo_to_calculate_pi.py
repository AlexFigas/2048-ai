"""
Use the montecarlo method to calculate pi

For this, plot one point at a time and calculate the value of pi
Per each iteration do the following:
    - Generate a random point (x, y) in the range [-1, 1]
    - Calculate the distance from the origin to the point
    - If the distance is less than 1, the point is inside the circle (plot it in red)
    - If the distance is greater than 1, the point is outside the circle (plot it in blue)
    - Calculate the value of pi as the ratio of the number of points inside the circle to the total number of points
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100000  # Number of iterations
PLOT_INTERVAL = 1000  # Interval to update the plot

# Initialize the plot
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_title("Monte Carlo method to calculate pi")

# Initialize the counters
inside_circle = 0

# Points storage for batch processing
inside_points = []
outside_points = []

# Perform Monte Carlo simulation
for i in range(1, N + 1):
    # Generate random point (x, y)
    x, y = 2 * np.random.rand(2) - 1  # Vectorized generation of (x, y)

    # Check if the point is inside the circle
    if x**2 + y**2 < 1:
        inside_circle += 1
        inside_points.append((x, y))
    else:
        outside_points.append((x, y))

    # Update the plot at specified intervals
    if i % PLOT_INTERVAL == 0 or i == N:
        ax.plot(
            [p[0] for p in inside_points],
            [p[1] for p in inside_points],
            "ro",
            markersize=1,
        )
        ax.plot(
            [p[0] for p in outside_points],
            [p[1] for p in outside_points],
            "bo",
            markersize=1,
        )
        inside_points, outside_points = [], []  # Clear the lists after plotting

        # Calculate and update pi
        pi = 4 * inside_circle / i
        ax.set_title(
            f"Monte Carlo method to calculate pi\nIteration: {i}\nPi: {pi:.4f}"
        )
        plt.pause(0.01)  # Reduced pause time for faster plotting

plt.show()

# Print the final value of pi
print(f"Final value of pi: {pi:.4f}")
