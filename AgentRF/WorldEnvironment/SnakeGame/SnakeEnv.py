import pygame
import time
import random
import torch

class envSnake:
    def __init__(self):
        # Initialize pygame
        pygame.init()

        # Set display dimensions
        self.object = 3
        self.objectD = {"background": 0,
                        "snake": 1,
                        "snack": 2}
        self.width, self.height = 600, 400
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")

        # Snake block size and speed
        self.block_size = 20
        self.snake_speed = 60*1

        # Info
        self.info = "Snake Game Environment"
        self.action_space = [4]
        self.observation_space = [self.object, self.width//self.block_size, self.height//self.block_size]

        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)

        # Fonts
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("comicsansms", 35)

        #Env
        # self.observation = torch.zeros((self.height, self.width, self.object),dtype=torch.half)
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.game_over = False
        self.game_close = False
        self.count_step = 0

        self.x = self.width / 2
        self.y = self.height / 2

        # self.dx = 0
        # self.dy = 0

        self.snake_list = []
        self.snake_length = 1

        self.snack_x = round(random.randrange(0, self.width - self.block_size) / 20.0) * 20.0
        self.snack_y = round(random.randrange(0, self.height - self.block_size) / 20.0) * 20.0
        # self.observation[self.snack_y, self.snack_x, self.objectD["snack"]] = 1

        snake_head = []
        snake_head.append(self.x)
        snake_head.append(self.y)
        self.snake_list.append(snake_head)
        # self.observation[self.y, self.x, self.objectD["snake"]] = 1

        self.clock = pygame.time.Clock()

        self.win.fill(self.black)
        pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

        self.draw_snake()
        self.score_display(self.snake_length - 1)
        pygame.display.update()


    def score_display(self, score):
        value = self.score_font.render("Score: " + str(score), True, self.white)
        self.win.blit(value, [0, 0])


    def draw_snake(self):
        for x in self.snake_list:
            pygame.draw.rect(self.win, self.green, [x[0], x[1], self.block_size, self.block_size])


    def message(self, msg, color):
        mesg = self.font_style.render(msg, True, color)
        self.win.blit(mesg, [self.width / 6, self.height / 3])

    
    def reset(self):
        # Reinit Env
        self.count_step = 0
        self.observation = torch.zeros((int(self.height//self.block_size), int(self.width//self.block_size), int(self.object)),dtype=torch.float)
        self.observation[:,:,0] = 1
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.game_over = False
        self.game_close = False
        self.direction = None


        self.x = self.width / 2
        self.y = self.height / 2

        # self.dx = 0
        # self.dy = 0

        self.snake_list = []
        self.snake_length = 1

        self.snack_x = round(random.randrange(0, self.width - self.block_size) / 20.0) * 20.0
        self.snack_y = round(random.randrange(0, self.height - self.block_size) / 20.0) * 20.0
        self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["snack"])] = 1
        self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["background"])] = 0

        snake_head = []
        snake_head.append(self.x)
        snake_head.append(self.y)
        self.snake_list.append(snake_head)
        self.observation[int(self.y//self.block_size), int(self.x//self.block_size), int(self.objectD["snake"])] = 1
        self.observation[int(self.y//self.block_size), int(self.x//self.block_size), int(self.objectD["background"])] = 0

        self.clock = pygame.time.Clock()

        self.win.fill(self.black)
        pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

        self.draw_snake()
        self.score_display(self.snake_length - 1)
        pygame.display.update()

        return self.observation, self.info

    def step(self,action):
        if self.count_step >= 50:
            self.truncated = True
            self.reward -= 2  # extra penalty for looping too long

        # Define directions
        LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
        opposites = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}

        # Prevent reversing direction
        if self.direction is not None and action == opposites[self.direction]:
            self.reward = -2  # Penalize hard
            self.terminated = True
            return self.observation, self.reward, self.terminated, self.truncated, self.info

        self.reward = 0
        if action == 0:
            dx = -self.block_size
            dy = 0
        elif action == 1:
            dx = self.block_size
            dy = 0
        elif action == 2:
            dy = -self.block_size
            dx = 0
        elif action == 3:
            dy = self.block_size
            dx = 0

        disbf = ((self.x - self.snack_x)**2 + (self.y - self.snack_y)**2)**(0.5)
        xMy = abs(self.x - self.snack_x) > abs(self.y - self.snack_y) and (dx != 0)
        yMx = abs(self.x - self.snack_x) < abs(self.y - self.snack_y) and (dy != 0)
        self.x += dx
        self.y += dy
        disaf = ((self.x - self.snack_x)**2 + (self.y - self.snack_y)**2)**(0.5)
        
        if disbf >= disaf:
            # self.reward = 0.1
            if xMy or yMx:
                self.reward = 0.1 #+ (1.0 / ((disbf + disaf)/2))*0.1
            else:
                self.reward = -0.05
            #     # print("test")
        else:
            self.reward = -0.05 #(-1.0 / ((disbf + disaf)/2))*0.1
            # print("inf")

        if self.x >= self.width or self.x < 0 or self.y >= self.height or self.y < 0:
            self.terminated = True
            self.reward = -5

        self.direction = action

        self.win.fill(self.black)
        pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

        snake_head = []
        snake_head.append(self.x)
        snake_head.append(self.y)
        self.snake_list.append(snake_head)

        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]

        for segment in self.snake_list[:-1]:
            if segment == snake_head:
                self.terminated = True
                self.reward = -5

        self.draw_snake()
        self.score_display(self.snake_length - 1)

        if self.x == self.snack_x and self.y == self.snack_y:
                
            while [self.snack_x,self.snack_y] in self.snake_list:
                self.snack_x = round(random.randrange(0, self.width - self.block_size) / 20.0) * 20.0
                self.snack_y = round(random.randrange(0, self.height - self.block_size) / 20.0) * 20.0
                
            pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

            self.snake_length += 1
            self.reward = 10
            self.count_step = 0

            if self.snake_length >= 30:
                self.truncated = True
        pygame.display.update()
        # self.clock.tick(self.snake_speed)
        if not(self.truncated or self.terminated):
            self.get_env()
        self.count_step += 1
        return self.observation, self.reward, self.terminated, self.truncated, self.info


    def get_env(self):
        self.observation = torch.zeros((int(self.height//self.block_size), int(self.width//self.block_size), int(self.object)),dtype=torch.float)
        self.observation[:,:,0] = 1
        self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["snack"])] = 1
        self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["background"])] = 0
        for x,y in self.snake_list:
            self.observation[int(y//self.block_size), int(x//self.block_size), int(self.objectD["snake"])] = 1
            self.observation[int(y//self.block_size), int(x//self.block_size), int(self.objectD["background"])] = 0
        return self.observation, self.reward, self.terminated, self.truncated