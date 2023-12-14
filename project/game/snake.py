import pygame
from pygame.surfarray import array3d
import numpy as np
import time
import random


class Colors:
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (213, 50, 80)
    green = (0, 255, 0)
    blue = (50, 153, 213)


class BaseGameObject:
    def __init__(self) -> None:
        self.color = Colors
        self.size = None
        self.x = None
        self.y = None

    def set_color(self, color: tuple):
        self.color = color

    def set_size(self, size: int):
        self.size = size


class Snake(BaseGameObject):
    def __init__(self, color=None, size=None, speed=None):
        super().__init__()
        self.color = color
        self.size = size
        self.speed = speed
        self.initial_length = 1
        self.length = self.initial_length
        self.score = 0
        self.head = []
        self.body = []

    def set_speed(self, speed):
        self.speed = speed

    def set_head(self, x, y):
        self.head = [x, y]


class Food(BaseGameObject):
    def __init__(self, color=None, size=None, width=640, height=480):
        super().__init__()
        self.color = color
        self.size = size
        self.width = width
        self.height = height
        self.randomize_coordinate()

    def randomize_coordinate(self):
        self.x = round(random.randrange(0, self.width - self.size) / 10.0) * 10.0
        self.y = round(random.randrange(0, self.height - self.size) / 10.0) * 10.0


class SnakeGame:
    pygame.init()
    game_display = pygame.display.set_mode((320, 240))

    def __init__(self) -> None:
        # Module On

        # 색상 정의
        self.color = Colors

        # 창 크기 설정
        self.fps = 120
        self.width, self.height = 320, 240

        self.snake_direction = (0, 0)
        # 타이틀 설정
        pygame.display.set_caption("Snake")

        self.clock = pygame.time.Clock()

        self.snake = Snake(color=self.color.black, size=10, speed=self.fps)
        self.food = Food(
            color=self.color.green, size=10, width=self.width, height=self.height
        )
        self.snake.x = self.width // 2
        self.snake.y = self.height // 2

        self.exit_game = False
        self.terminal = False

    def hit_wall(self):
        if (
            self.snake.x < 0
            or self.snake.y < 0
            or self.snake.x >= self.width
            or self.snake.y >= self.height
        ):
            return True
        else:
            return False

    def snake_state(self):
        for x in self.snake.body:
            pygame.draw.rect(
                self.game_display,
                self.color.black,
                [x[0], x[1], self.snake.size, self.snake.size],
            )

    def food_gen(self):
        pygame.draw.rect(
            self.game_display,
            self.food.color,
            [self.food.x, self.food.y, self.food.size, self.food.size],
        )

    def compute_delta_by_action(self, action):
        delta = {
            0: (-self.snake.size, 0),  # Left
            1: (self.snake.size, 0),  # Right
            2: (0, -self.snake.size),  # Up
            3: (0, self.snake.size),  # Down
        }
        return delta.get(action, (0, 0))

    def frame(self, action=None):
        terminal = False
        reward = 0.0
        self.game_display.fill(self.color.white)
        self.food_gen()

        x_delta, y_delta = self.snake_direction

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit_game = True
        x_delta = 0
        y_delta = 0

        x_delta, y_delta = self.compute_delta_by_action(action)

        self.snake.x += x_delta
        self.snake.y += y_delta

        self.snake.head = [self.snake.x, self.snake.y]
        self.snake.body.append(self.snake.head)

        if len(self.snake.body) > self.snake.length:
            del self.snake.body[0]

        if self.hit_wall() or self.snake.head in self.snake.body[:-1]:
            terminal = True
            reward = -1.0

        self.snake_state()
        if self.snake.x == self.food.x and self.snake.y == self.food.y:
            self.snake.length += 1
            reward = 1.0
            self.food.randomize_coordinate()

        if terminal:
            self.__init__()

        image = array3d(pygame.display.get_surface())
        pygame.display.update()
        self.clock.tick(self.fps)
        return image, reward, terminal

    def gameLoop(self):
        while not self.exit_game:
            _, _, terminal = self.frame()
            self.terminal = terminal
            while self.terminal:
                self.terminal_display()

        pygame.quit()
        quit()


def main():
    game = SnakeGame()
    game.game_start()


if __name__ == "__main__":
    main()
