import numpy as np
from Game2048 import Game2048
import copy


class MonteCarloAgent:
    def initialize_game(self):
        return Game2048()

    def random_move(self, game):
        moves = game.get_valid_moves()
        if moves:
            return np.random.choice(moves)
        return None

    def simulate(self, game, search_length):
        simulated_game = copy.deepcopy(
            game
        )  # Deep copy to not affect the original game
        total_score = 0

        for _ in range(search_length):
            move = self.random_move(simulated_game)
            if move:
                simulated_game.move(move)
                total_score += simulated_game.get_max_tile()
            else:
                break  # No valid move, game over

        return total_score

    def move(self, game, searches_per_move, search_length):
        first_moves = game.get_valid_moves()
        scores = np.zeros(len(first_moves))

        for i, move in enumerate(first_moves):
            total_score = 0

            for _ in range(searches_per_move):
                # Clone the game to not affect the original board
                cloned_game = copy.deepcopy(game)
                cloned_game.move(move)
                total_score += self.simulate(cloned_game, search_length)

            scores[i] = total_score

        best_move_index = np.argmax(scores)
        best_move = first_moves[best_move_index]
        game.move(best_move)

        return game.get_state(), best_move


# Test the MonteCarloAgent class
agent = MonteCarloAgent()
game = agent.initialize_game()

number_of_searches = 40
search_length = 10

while not game.is_game_over():
    final_board, best_move = agent.move(game, number_of_searches, search_length)
    print(f"Best move: {best_move}")
    print(f"Final board:\n{final_board}")

if game.is_win():
    print("You won!")
else:
    print("Game over!")
