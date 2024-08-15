import sys
import os
import numpy as np
from Game2048 import Game2048, TILE_COLORS
import matplotlib.pyplot as plt
from AdvancedMonteCarloAgent import AdvancedMonteCarloAgent
from tqdm import tqdm
from joblib import Parallel, delayed
import time


def suppress_pygame_output(func):
    """Decorator to suppress stdout and stderr temporarily."""

    def wrapper(*args, **kwargs):
        # Suppress output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        try:
            result = func(*args, **kwargs)
        finally:
            # Restore output
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        return result

    return wrapper


@suppress_pygame_output
def run_single_simulation(agent):
    game = Game2048()
    game.reset()
    while not game.is_game_over():
        agent.move(game, 40, 10)
    win = game.is_win()
    board = game.get_state()
    unique, counts = np.unique(board, return_counts=True)
    tile_counts = dict(zip(unique, counts))
    max_tile = max(tile_counts.keys())
    score = game.get_score()

    return win, max_tile, score


def run_simulations_in_parallel(agent, num_simulations):
    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(agent) for _ in tqdm(range(num_simulations))
    )
    win_count = sum(1 for result in results if result[0])

    tile_distribution = {
        131072: 0,
        65536: 0,
        32768: 0,
        16384: 0,
        8192: 0,
        4096: 0,
        2048: 0,
        1024: 0,
        512: 0,
        256: 0,
        128: 0,
        64: 0,
        32: 0,
        16: 0,
        8: 0,
        4: 0,
        2: 0,
    }

    for _, max_tile, _ in results:
        tile_distribution[max_tile] += 1

    win_percentage = (win_count / num_simulations) * 100

    best_score = max(results, key=lambda x: x[2])[2]

    return win_percentage, tile_distribution, best_score


# Function to normalize RGB values to [0, 1] range
def normalize_rgb(color):
    return tuple(c / 255 for c in color)


def plot_tile_distribution(tile_distribution, win_percentage, best_score):
    """Plot the tile distribution with values on top of each bar using specific colors."""
    sorted_tiles = sorted(tile_distribution.keys())
    sorted_counts = [tile_distribution[tile] for tile in sorted_tiles]

    plt.figure(figsize=(12, 6))
    bar_width = 0.8
    x_positions = np.arange(len(sorted_tiles))

    # Assign colors based on TILE_COLORS
    bar_colors = [normalize_rgb(TILE_COLORS[tile]) for tile in sorted_tiles]

    bars = plt.bar(
        x_positions, sorted_counts, width=bar_width, color=bar_colors, align="center"
    )

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval, int(yval), ha="center", va="bottom"
        )

    plt.xticks(x_positions, sorted_tiles, rotation=45)
    plt.xlabel("Tile Value")
    plt.ylabel("Frequency")
    plt.title(
        "Tile Distribution (Win Percentage: {:.2f}%, Best Score: {})".format(
            win_percentage, best_score
        )
    )
    plt.tight_layout()

    # Save the figure
    plt.savefig("images/{}_{}.png".format(best_score, win_percentage))
    # plt.show(block=False)


def main():
    num_simulations = 500  # Adjust the number of simulations as needed
    agent = AdvancedMonteCarloAgent()

    win_percentage, tile_distribution, best_score = run_simulations_in_parallel(
        agent, num_simulations
    )

    print("Win Percentage: {:.0f}%".format(np.round(win_percentage)))
    print("Tile Distribution: ", tile_distribution)

    # Call the plot function
    plot_tile_distribution(tile_distribution, win_percentage, best_score)


if __name__ == "__main__":
    main()
