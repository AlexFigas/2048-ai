import os
import sys

# Suppress stdout and stderr
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

import copy
import numpy as np
import pygame
from Game2048 import Game2048
from Game2048 import (
    WIDTH,
    HEIGHT,
)
from joblib import Parallel, delayed
import time

# Restore stdout and stderr after imports
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


class AdvancedMonteCarloAgent:
    def random_move(self, game):
        moves = game.get_valid_moves()
        if moves:
            return np.random.choice(moves)
        return None

    def heuristic_score(self, game):
        board = game.get_state()
        empty_cells = np.sum(board == 0)
        max_tile = np.max(board)

        def monotonicity(board):
            def check_line(line):
                score = 0
                for i in range(1, len(line)):
                    if line[i] > line[i - 1]:
                        score += line[i] - line[i - 1]
                    elif line[i] < line[i - 1]:
                        score += line[i - 1] - line[i]
                return score

            score = 0
            for i in range(4):
                score += check_line(board[i, :]) + check_line(board[:, i])

            return score

        def smoothness(board):
            score = 0
            for i in range(4):
                for j in range(4):
                    if board[i, j] == 0:
                        continue
                    value = np.log2(board[i, j])
                    for direction in [(0, 1), (1, 0)]:
                        next_i, next_j = i + direction[0], j + direction[1]
                        if (
                            0 <= next_i < 4
                            and 0 <= next_j < 4
                            and board[next_i, next_j] > 0
                        ):
                            next_value = np.log2(board[next_i, next_j])
                            score -= abs(value - next_value)
            return score

        def weighted_tiles(board):
            weight_matrix = np.array(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    # [7, 6, 5, 4],
                    # [6, 5, 4, 3],
                    # [5, 4, 3, 2],
                    # [4, 3, 2, 1]
                    # [4**6, 4**5, 4**4, 4**3],
                    # [4**5, 4**4, 4**3, 4**2],
                    # [4**4, 4**3, 4**2, 4**1],
                    # [4**3, 4**2, 4**1, 4**0]
                    # [4**15, 4**14, 4**13, 4**12],
                    # [4**8, 4**9, 4**10, 4**11],
                    # [4**7, 4**6, 4**5, 4**4],
                    # [4**0, 4**1, 4**2, 4**3]
                ]
            )
            score = np.sum(weight_matrix * board)
            return score

        def clustering_penalty(board):
            penalty = 0
            for i in range(4):
                for j in range(4):
                    if board[i, j] > 0:
                        penalty += min(
                            [
                                abs(board[i, j] - board[i + di, j + dj])
                                for di in [-1, 0, 1]
                                for dj in [-1, 0, 1]
                                if 0 <= i + di < 4
                                and 0 <= j + dj < 4
                                and (di != 0 or dj != 0)
                            ]
                        )
            return penalty

        # Weighted sum of heuristic components
        heuristic = (
            (0.4 * empty_cells)
            + (1.2 * max_tile)
            - (0.2 * monotonicity(board))
            + (0.3 * smoothness(board))
            + (0.5 * weighted_tiles(board))
            - (0.4 * clustering_penalty(board))
        )

        return heuristic

    def simulate(self, game, search_length):
        simulated_game = copy.deepcopy(game)
        total_score = 0

        for _ in range(search_length):
            move = self.random_move(simulated_game)
            if move:
                simulated_game.move(move)
                total_score += self.heuristic_score(simulated_game)
            else:
                break  # No valid move, game over

        return total_score

    def move(self, game, searches_per_move, search_length):
        first_moves = game.get_valid_moves()
        scores = np.zeros(len(first_moves))

        def simulate_move(move):
            total_score = 0
            for _ in range(searches_per_move):
                cloned_game = copy.deepcopy(game)
                cloned_game.move(move)
                score = self.simulate(cloned_game, search_length)
                total_score += score
            return total_score

        # Parallelize the move simulations
        scores = Parallel(n_jobs=-1)(
            delayed(simulate_move)(move) for move in first_moves
        )

        # Prioritize moves leading to higher scores
        best_move_index = np.argmax(scores)
        best_move = first_moves[best_move_index]
        game.move(best_move)

        return game.get_state(), best_move


# Main loop with AI playing
def main():
    # Initialize game and agent
    game = Game2048()
    agent = AdvancedMonteCarloAgent()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048 AI")
    game.reset()

    number_of_searches = 40  # Increase the number of searches for better decisions
    search_length = 10  # Simulate longer sequences of moves
    win_shown = False  # Flag to check if winning message has been shown

    while not game.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # AI plays the move
        final_board, best_move = agent.move(game, number_of_searches, search_length)
        print(f"Best move: {best_move}")
        print(f"Final board:\n{final_board}")

        if game.is_win() and not win_shown:
            win_shown = True  # Set the flag to True after detecting the win

        # Draw the board after AI move
        game.draw_board(screen)

    if game.is_game_over():
        if win_shown:
            print("Game over after winning!")
        else:
            print("Game over!")

    print(game.get_score())


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
