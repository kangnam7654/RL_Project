import pygame
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
    
    def set_color(self, color: tuple):
        self.color = color

    def set_size(self, size: int):
        self.size = size
        
class Snake(BaseGameObject):
    def __init__(self, color=(0, 0, 0), size=10, speed=15):
        super().__init__()
        self.color = color
        self.size = size
        self.speed = speed
        self.initial_length = 1
        self.head = []
        self.body = []

    def set_speed(self, speed):
        self.speed = speed

class Food(BaseGameObject):
    def __init__(self):
        self.color = 

class Colors:
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (213, 50, 80)
    green = (0, 255, 0)
    blue = (50, 153, 213)

class SnakeGame:
    def __init__(self) -> None:
        pygame.init()

        # 색상 정의
        self.color = Colors

        # 창 크기 설정
        self.width, self.height = 600, 400
        self.game_display = pygame.display.set_mode((self.width, self.height))

        # 타이틀 설정
        pygame.display.set_caption("Snake")

        self.clock = pygame.time.Clock()

        self.snake = Snake()
        self.font_style = pygame.font.SysFont(None, 50)
        self.game_over = False
        self.game_close = False

    def our_snake(self, snake_block, snake_list):
        for x in snake_list:
            pygame.draw.rect(
                self.game_display,
                self.color.black,
                [x[0], x[1], snake_block, snake_block],
            )

    def message(self, msg, color):
        mesg = self.font_style.render(msg, True, color)
        self.game_display.blit(mesg, [self.width / 6, self.height / 3])

    def gameLoop(self):
        x1 = self.width // 2
        y1 = self.height // 2

        x1_change = 0
        y1_change = 0

        foodx = round(random.randrange(0, self.width - self.snake.size) / 10.0) * 10.0
        foody = round(random.randrange(0, self.height - self.snake.size) / 10.0) * 10.0

        while not self.game_over:
            while self.game_close:  # Game Over == False, Game Close == True
                self.game_display.fill(self.color.white)
                self.message("Lose! C-Restart or Q-Quit", self.color.red)
                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.game_over = True
                            self.game_close = False
                        if event.key == pygame.K_c:
                            self.gameLoop()

            for event in pygame.event.get():  # False False
                if event.type == pygame.QUIT:
                    self.game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        x1_change = -self.snake_block
                        y1_change = 0
                    elif event.key == pygame.K_RIGHT:
                        x1_change = self.snake_block
                        y1_change = 0
                    elif event.key == pygame.K_UP:
                        y1_change = -self.snake_block
                        x1_change = 0
                    elif event.key == pygame.K_DOWN:
                        y1_change = self.snake_block
                        x1_change = 0

            if x1 >= self.width or x1 < 0 or y1 >= self.height or y1 < 0:
                self.game_close = True

            x1 += x1_change
            y1 += y1_change
            self.game_display.fill(self.white)
            pygame.draw.rect(
                self.game_display,
                self.color.green,
                [foodx, foody, self.snake_block, self.snake_block],
            )
            snake_head = []
            snake_head.append(x1)
            snake_head.append(y1)
            self.snake.snake_body.append(snake_head)
            if len(self.snake.snake_body) > self.snake.initial_length:
                del self.snake.snake_body[0]

            for x in self.snake.snake_body[:-1]:
                if x == snake_head:
                    self.game_close = True

            self.our_snake(self.snake.size, self.snake.snake_body)
            pygame.display.update()

            if x1 == foodx and y1 == foody:
                foodx = (
                    round(random.randrange(0, self.width - self.snake_block) / 10.0)
                    * 10.0
                )
                foody = (
                    round(random.randrange(0, self.height - self.snake_block) / 10.0)
                    * 10.0
                )
                length_of_snake += 1

            self.clock.tick(self.snake.speed)

        pygame.quit()
        quit()

if __name__ == "__main__":
    pygame.init()
