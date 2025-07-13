import pygame
import time
import random

# Initialize pygame
pygame.init()

# Set display dimensions
width, height = 600, 400
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Snake block size and speed
block_size = 20
snake_speed = 5

# Fonts
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)


def score_display(score):
    value = score_font.render("Score: " + str(score), True, white)
    win.blit(value, [0, 0])


def draw_snake(block_size, snake_list):
    for x in snake_list:
        pygame.draw.rect(win, green, [x[0], x[1], block_size, block_size])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    win.blit(mesg, [width / 6, height / 3])


def game_loop():
    game_over = False
    game_close = False

    x = width / 2
    y = height / 2

    dx = 0
    dy = 0

    snake_list = []
    snake_length = 1

    snack_x = round(random.randrange(0, width - block_size) / 20.0) * 20.0
    snack_y = round(random.randrange(0, height - block_size) / 20.0) * 20.0

    clock = pygame.time.Clock()

    while not game_over:

        while game_close:
            win.fill(blue)
            message("You Lost! Press Q-Quit or C-Play Again", red)
            score_display(snake_length - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    dx = -block_size
                    dy = 0
                elif event.key == pygame.K_RIGHT:
                    dx = block_size
                    dy = 0
                elif event.key == pygame.K_UP:
                    dy = -block_size
                    dx = 0
                elif event.key == pygame.K_DOWN:
                    dy = block_size
                    dx = 0

        x += dx
        y += dy

        if x >= width or x < 0 or y >= height or y < 0:
            game_close = True

        win.fill(black)
        pygame.draw.rect(win, red, [snack_x, snack_y, block_size, block_size])

        snake_head = []
        snake_head.append(x)
        snake_head.append(y)
        snake_list.append(snake_head)

        if len(snake_list) > snake_length:
            del snake_list[0]

        for segment in snake_list[:-1]:
            if segment == snake_head:
                game_close = True

        draw_snake(block_size, snake_list)
        score_display(snake_length - 1)

        pygame.display.update()

        if x == snack_x and y == snack_y:
            while [snack_x, snack_y] in snake_list:
                snack_x = round(random.randrange(0, width - block_size) / 20.0) * 20.0
                snack_y = round(random.randrange(0, height - block_size) / 20.0) * 20.0
            snake_length += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()


game_loop()
