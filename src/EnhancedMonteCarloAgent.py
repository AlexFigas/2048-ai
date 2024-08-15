import numpy as np
from Game2048 import Game2048
import copy
from joblib import Parallel, delayed


class EnhancedMonteCarloAgent:
    def initialize_game(self):
        return Game2048()

    def random_move(self, game):
        moves = game.get_valid_moves()
        if moves:
            return np.random.choice(moves)
        return None

    def heuristic_score(self, game):
        board = game.get_state()
        empty_cells = np.sum(board == 0)
        max_tile = np.max(board)
        return empty_cells + max_tile

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

        best_move_index = np.argmax(scores)
        best_move = first_moves[best_move_index]
        game.move(best_move)

        return game.get_state(), best_move


# Test the EnhancedMonteCarloAgent class
agent = EnhancedMonteCarloAgent()
game = agent.initialize_game()

number_of_searches = 40  # Increase the number of searches for better decisions
search_length = 10  # Simulate longer sequences of moves

while not game.is_game_over():
    final_board, best_move = agent.move(game, number_of_searches, search_length)
    print(f"Best move: {best_move}")
    print(f"Final board:\n{final_board}")
    if game.is_win():
        print("You won!")
        break

if game.is_game_over() and not game.is_win():
    print("Game over!")
