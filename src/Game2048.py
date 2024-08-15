import os
import sys

sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

import numpy as np
import random
import pygame

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Colors
BACKGROUND_COLOR = (30, 30, 30)
EMPTY_TILE_COLOR = (30, 30, 30)
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (181, 134, 180),
    8192: (168, 97, 171),
    16384: (160, 72, 163),
    32768: (128, 0, 128),
    65536: (96, 0, 70),
    131072: (139, 134, 227),
}
FONT_COLOR = (119, 110, 101)

# Initialize Pygame
pygame.init()

# Game Constants
SIZE = 4
WIDTH = 400
HEIGHT = 400
TILE_SIZE = WIDTH // SIZE
FONT = pygame.font.SysFont("arial", 24)
BOLD_FONT = pygame.font.SysFont("arial", 24, bold=True)


class Game2048:
    def __init__(self):
        self.reset()
        self.score = 0

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_new_tile()
        self.add_new_tile()
        self.score = 0
        return self.board

    def add_new_tile(self):
        empty_cells = list(zip(*np.nonzero(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def slide_and_merge(self, row):
        new_row = np.zeros_like(row)
        previous = 0
        new_index = 0
        score_increment = 0

        for i in range(len(row)):
            if row[i] != 0:
                if previous == row[i]:
                    new_row[new_index - 1] *= 2
                    score_increment += new_row[new_index - 1]
                    previous = 0
                else:
                    previous = row[i]
                    new_row[new_index] = row[i]
                    new_index += 1

        return new_row, score_increment

    def move(self, direction):
        moved = False
        total_score_increment = 0

        for i in range(4):
            if direction == "left":
                new_row, score_increment = self.slide_and_merge(self.board[i])
                if not np.array_equal(self.board[i], new_row):
                    moved = True
                    self.board[i] = new_row
                    total_score_increment += score_increment
            elif direction == "right":
                new_row, score_increment = self.slide_and_merge(self.board[i][::-1])
                new_row = new_row[::-1]
                if not np.array_equal(self.board[i], new_row):
                    moved = True
                    self.board[i] = new_row
                    total_score_increment += score_increment
            elif direction == "up":
                new_row, score_increment = self.slide_and_merge(self.board[:, i])
                if not np.array_equal(self.board[:, i], new_row):
                    moved = True
                    self.board[:, i] = new_row
                    total_score_increment += score_increment
            elif direction == "down":
                new_row, score_increment = self.slide_and_merge(self.board[:, i][::-1])
                new_row = new_row[::-1]
                if not np.array_equal(self.board[:, i], new_row):
                    moved = True
                    self.board[:, i] = new_row
                    total_score_increment += score_increment

        if moved:
            self.add_new_tile()
            self.score += total_score_increment

        return moved

    def get_valid_moves(self):
        valid_moves = []
        for direction in ["left", "right", "up", "down"]:
            copy_board = self.board.copy()
            copy_score = self.score
            if self.move(direction):
                valid_moves.append(direction)
                self.board = copy_board
                self.score = copy_score
        return valid_moves

    def is_game_over(self):
        return len(self.get_valid_moves()) == 0

    def get_state(self):
        return self.board.copy()

    def move_left(self):
        return self.move("left")

    def move_right(self):
        return self.move("right")

    def move_up(self):
        return self.move("up")

    def move_down(self):
        return self.move("down")

    def is_win(self):
        return np.any(self.board >= 2048)

    def get_max_tile(self):
        return np.max(self.board)

    def draw_board(self, screen):
        screen.fill(BACKGROUND_COLOR)
        for i in range(SIZE):
            for j in range(SIZE):
                value = self.board[i][j]
                color = TILE_COLORS.get(value, EMPTY_TILE_COLOR)
                pygame.draw.rect(
                    screen, color, (j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                )
                if value != 0:
                    text_surface = BOLD_FONT.render(
                        str(value), True, FONT_COLOR if value <= 4 else (255, 255, 255)
                    )
                    text_rect = text_surface.get_rect(
                        center=(
                            j * TILE_SIZE + TILE_SIZE / 2,
                            i * TILE_SIZE + TILE_SIZE / 2,
                        )
                    )
                    screen.blit(text_surface, text_rect)

        self.draw_winning_message(screen) if self.is_win() else None
        pygame.display.flip()

    def draw_winning_message(self, screen):
        # Create a semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(200)  # Set transparency level (0-255)
        overlay.fill((0, 0, 0))  # Fill it with black color
        screen.blit(overlay, (0, 0))

        # Render the winning text
        winning_text = BOLD_FONT.render("You won!", True, (255, 255, 255))
        text_rect = winning_text.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        screen.blit(winning_text, text_rect)

    def get_score(self):
        return self.score


# Main loop
def main():
    # Initialize game
    game = Game2048()

    # Suppress stdout
    sys.stdout = open(os.devnull, "w")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048")
    game.reset()

    while not game.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move("left")
                elif event.key == pygame.K_RIGHT:
                    game.move("right")
                elif event.key == pygame.K_UP:
                    game.move("up")
                elif event.key == pygame.K_DOWN:
                    game.move("down")

        game.draw_board(screen)

    # Restore stdout
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
