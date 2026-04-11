# import pygame
# import time
# import random
# import torch
# from os import environ
# environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# class envSnake:
#     def __init__(self):
#         # Initialize pygame
#         pygame.init()

#         # Set display dimensions
#         self.object = 3
#         self.objectD = {"background": 0,
#                         "snake": 1,
#                         "snack": 2}
#         self.width, self.height = 600, 400
#         self.win = pygame.display.set_mode((self.width, self.height))
#         pygame.display.set_caption("Snake Game")

#         # Snake block size and speed
#         self.block_size = 20
#         self.snake_speed = 60*1

#         # Info
#         self.info = "Snake Game Environment"
#         self.action_space = [4]
#         self.observation_space = [self.object, self.width//self.block_size, self.height//self.block_size]

#         # Colors
#         self.white = (255, 255, 255)
#         self.black = (0, 0, 0)
#         self.red = (213, 50, 80)
#         self.green = (0, 255, 0)
#         self.blue = (50, 153, 213)

#         # Fonts
#         self.font_style = pygame.font.SysFont("bahnschrift", 25)
#         self.score_font = pygame.font.SysFont("comicsansms", 35)

#         #Env
#         # self.observation = torch.zeros((self.height, self.width, self.object),dtype=torch.half)
#         self.reward = 0
#         self.terminated = False
#         self.truncated = False
#         self.game_over = False
#         self.game_close = False
#         self.count_step = 0

#         self.x = self.width / 2
#         self.y = self.height / 2

#         # self.dx = 0
#         # self.dy = 0

#         self.snake_list = []
#         self.snake_length = 1

#         self.snack_x = round(random.randrange(0, self.width - self.block_size) / 20.0) * 20.0
#         self.snack_y = round(random.randrange(0, self.height - self.block_size) / 20.0) * 20.0
#         # self.observation[self.snack_y, self.snack_x, self.objectD["snack"]] = 1

#         snake_head = []
#         snake_head.append(self.x)
#         snake_head.append(self.y)
#         self.snake_list.append(snake_head)
#         # self.observation[self.y, self.x, self.objectD["snake"]] = 1

#         self.clock = pygame.time.Clock()

#         self.win.fill(self.black)
#         pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

#         self.draw_snake()
#         self.score_display(self.snake_length - 1)
#         pygame.display.update()


#     def score_display(self, score):
#         value = self.score_font.render("Score: " + str(score), True, self.white)
#         self.win.blit(value, [0, 0])


#     def draw_snake(self):
#         for x in self.snake_list:
#             pygame.draw.rect(self.win, self.green, [x[0], x[1], self.block_size, self.block_size])


#     def message(self, msg, color):
#         mesg = self.font_style.render(msg, True, color)
#         self.win.blit(mesg, [self.width / 6, self.height / 3])

    
#     def reset(self):
#         # Reinit Env
#         self.count_step = 0
#         self.observation = torch.zeros((int(self.height//self.block_size), int(self.width//self.block_size), int(self.object)),dtype=torch.float)
#         self.observation[:,:,0] = 1
#         self.reward = 0
#         self.score = 0
#         self.terminated = False
#         self.truncated = False
#         self.game_over = False
#         self.game_close = False
#         self.direction = None


#         self.x = self.width / 2
#         self.y = self.height / 2

#         # self.dx = 0
#         # self.dy = 0

#         self.snake_list = []
#         self.snake_length = 1

#         self.snack_x = round(random.randrange(0, self.width - self.block_size) / 20.0) * 20.0
#         self.snack_y = round(random.randrange(0, self.height - self.block_size) / 20.0) * 20.0
#         self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["snack"])] = 1
#         self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["background"])] = 0

#         snake_head = []
#         snake_head.append(self.x)
#         snake_head.append(self.y)
#         self.snake_list.append(snake_head)
#         self.observation[int(self.y//self.block_size), int(self.x//self.block_size), int(self.objectD["snake"])] = 1
#         self.observation[int(self.y//self.block_size), int(self.x//self.block_size), int(self.objectD["background"])] = 0

#         self.clock = pygame.time.Clock()

#         self.win.fill(self.black)
#         pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

#         self.draw_snake()
#         self.score_display(self.snake_length - 1)
#         pygame.display.update()

#         return self.observation, self.info

#     def step(self,action):
#         if self.count_step >= 50:
#             self.truncated = True
#             self.reward -= 0  # extra penalty for looping too long

#         # Define directions
#         LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
#         opposites = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}

#         # Prevent reversing direction
#         if self.direction is not None and action == opposites[self.direction]:
#             self.reward = -2  # Penalize hard
#             self.terminated = True
#             return self.observation, self.reward, self.terminated, self.truncated, self.score, self.info

#         self.reward = 0
#         if action == 0:
#             dx = -self.block_size
#             dy = 0
#         elif action == 1:
#             dx = self.block_size
#             dy = 0
#         elif action == 2:
#             dy = -self.block_size
#             dx = 0
#         elif action == 3:
#             dy = self.block_size
#             dx = 0

#         disbf = ((self.x - self.snack_x)**2 + (self.y - self.snack_y)**2)**(0.5)
#         xMy = abs(self.x - self.snack_x) > abs(self.y - self.snack_y) and (dx != 0)
#         yMx = abs(self.x - self.snack_x) < abs(self.y - self.snack_y) and (dy != 0)
#         self.x += dx
#         self.y += dy
#         disaf = ((self.x - self.snack_x)**2 + (self.y - self.snack_y)**2)**(0.5)
        
#         if disbf >= disaf:
#             # self.reward = 0.1
#             if xMy or yMx:
#                 self.reward = 0.1 #+ (1.0 / ((disbf + disaf)/2))*0.1
#             else:
#                 self.reward = -0.1
#             #     # print("test")
#         else:
#             self.reward = -0.1 #(-1.0 / ((disbf + disaf)/2))*0.1
#             # print("inf")

#         if self.x >= self.width or self.x < 0 or self.y >= self.height or self.y < 0:
#             self.terminated = True
#             self.reward = -5

#         self.direction = action

#         self.win.fill(self.black)
#         pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

#         snake_head = []
#         snake_head.append(self.x)
#         snake_head.append(self.y)
#         self.snake_list.append(snake_head)

#         if len(self.snake_list) > self.snake_length:
#             del self.snake_list[0]

#         for segment in self.snake_list[:-1]:
#             if segment == snake_head:
#                 self.terminated = True
#                 self.reward = -5

#         self.draw_snake()
#         self.score_display(self.snake_length - 1)

#         if self.x == self.snack_x and self.y == self.snack_y:
                
#             while [self.snack_x,self.snack_y] in self.snake_list:
#                 self.snack_x = round(random.randrange(0, self.width - self.block_size) / 20.0) * 20.0
#                 self.snack_y = round(random.randrange(0, self.height - self.block_size) / 20.0) * 20.0
                
#             pygame.draw.rect(self.win, self.red, [self.snack_x, self.snack_y, self.block_size, self.block_size])

#             self.snake_length += 1
#             self.reward = 5
#             self.count_step = 0
#             self.score += 1

#             if self.snake_length >= 30:
#                 self.truncated = True
#         pygame.display.update()
#         # self.clock.tick(self.snake_speed)
#         if not(self.truncated or self.terminated):
#             self.get_env()
#         self.count_step += 1
#         return self.observation, self.reward, self.terminated, self.truncated, self.score, self.info


#     def get_env(self):
#         self.observation = torch.zeros((int(self.height//self.block_size), int(self.width//self.block_size), int(self.object)),dtype=torch.float)
#         self.observation[:,:,0] = 1
#         self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["snack"])] = 1
#         self.observation[int(self.snack_y//self.block_size), int(self.snack_x//self.block_size), int(self.objectD["background"])] = 0
#         for x,y in self.snake_list:
#             self.observation[int(y//self.block_size), int(x//self.block_size), int(self.objectD["snake"])] = 1
#             self.observation[int(y//self.block_size), int(x//self.block_size), int(self.objectD["background"])] = 0
#         return self.observation, self.reward, self.terminated, self.truncated, self.score
    
#     def close(self):
#         pygame.quit()



import numpy as np
import random
# เอา pygame ออกจากการคำนวณหลัก ถ้าไม่จำเป็นต้องเรนเดอร์

class envSnake:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.width, self.height = 600, 400
        self.block_size = 20
        self.grid_w = self.width // self.block_size
        self.grid_h = self.height // self.block_size
        
        self.object = 4  # เพิ่มเป็น 4 object
        self.objectD = {"background": 0, "snake_body": 1, "snack": 2, "snake_head": 3}
        
        self.action_space = [4]
        self.observation_space = [self.object, self.grid_h, self.grid_w]
        self.info = "Snake Game Environment"

        # --- PyGame Setup (ทำเฉพาะตอนต้องการดูภาพ) ---
        if self.render_mode:
            import pygame
            pygame.init()
            self.win = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
            self.colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (213, 50, 80), "green": (0, 255, 0)}
            self.font = pygame.font.SysFont("comicsansms", 35)

    def reset(self):
        self.count_step = 0
        # ยืดหยุ่นเวลาขึ้นอยู่กับความยาวงู
        self.max_steps = 2048
        self.bias_step =  64# จุดเริ่มต้นของการให้รางวัลติดลบเพื่อกระตุ้นให้หาทางออกเร็วขึ้น
        
        self.reward = 0
        self.score = 0
        self.terminated = False
        self.truncated = False
        self.direction = None

        # ใช้พิกัดกริด (0-29, 0-19) แทนพิกัด Pixel จะคำนวณเร็วกว่า
        self.x = self.grid_w // 2
        self.y = self.grid_h // 2

        self.snake_list = [[self.x, self.y]]
        self.snake_length = 1

        self._place_snack()
        
        if self.render_mode:
            self._render_frame()

        return self.get_env(), self.info

    def _place_snack(self):
        while True:
            self.snack_x = random.randrange(0, self.grid_w)
            self.snack_y = random.randrange(0, self.grid_h)
            if [self.snack_x, self.snack_y] not in self.snake_list:
                break

    def step(self, action):
        if self.count_step >= self.max_steps:
            self.truncated = True
        
        if self.count_step % self.bias_step == 0 and self.count_step > 0:
            self.reward -= 0.1  # extra penalty for looping too long

        LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
        opposites = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}

        # การกันงูเดินถอยหลัง
        if self.direction is not None and action == opposites.get(self.direction):
            # หักคะแนนรุนแรงและจบเกม
            self.reward = -2
            self.terminated = True
            return self.get_env(), self.reward, self.terminated, self.truncated, self.score, self.info

        dx, dy = 0, 0
        if action == 0: dx = -1
        elif action == 1: dx = 1
        elif action == 2: dy = -1
        elif action == 3: dy = 1

        # คำนวณระยะทางเพื่อ Reward Shaping
        disbf = abs(self.x - self.snack_x) + abs(self.y - self.snack_y) # Manhattan Distance เหมาะกับ Grid มากกว่า Euclidean

        self.x += dx
        self.y += dy
        self.direction = action
        
        disaf = abs(self.x - self.snack_x) + abs(self.y - self.snack_y)

        # การให้รางวัลพื้นฐาน
        if disbf > disaf:
            self.reward = 0.03
        else:
            self.reward = -0.05

        # self.reward = -0.01 # ให้รางวัลติดลบเล็กน้อยทุกก้าวเพื่อกระตุ้นให้หาทางออกเร็วขึ้น

        # เช็คชนกำแพง
        if self.x >= self.grid_w or self.x < 0 or self.y >= self.grid_h or self.y < 0:
            self.terminated = True
            self.reward = -2
            return self.get_env(), self.reward, self.terminated, self.truncated, self.score, self.info

        snake_head = [self.x, self.y]
        
        # เช็คชนตัวเอง
        if snake_head in self.snake_list:
            self.terminated = True
            self.reward = -4
            return self.get_env(), self.reward, self.terminated, self.truncated, self.score, self.info

        self.snake_list.append(snake_head)

        # จัดการการกินอาหาร
        if self.x == self.snack_x and self.y == self.snack_y:
            self.snake_length += 1
            self.reward = 1 # ให้รางวัลมากขึ้นเมื่อกินได้
            self.score += 1
            self.count_step = 0 # รีเซ็ตตัวนับเมื่อกินได้
            self.max_steps += 10 # เพิ่มเวลาให้อยู่รอดได้นานขึ้นเมื่องูยาวขึ้น
            self._place_snack()
        else:
            # ถ้าไม่ได้กิน ต้องตัดหางทิ้ง
            if len(self.snake_list) > self.snake_length:
                del self.snake_list[0]

        self.count_step += 1
        
        if self.render_mode:
            self._render_frame()

        return self.get_env(), self.reward, self.terminated, self.truncated, self.score, self.info

    def get_env(self):
        # ใช้ Numpy สร้าง Array ให้เบาและเร็วที่สุด
        obs = np.zeros((self.grid_h, self.grid_w, self.object), dtype=np.float32)
        obs[:, :, 0] = 1.0 # Background
        
        # ใส่ Snack
        obs[self.snack_y, self.snack_x, self.objectD["snack"]] = 1.0
        obs[self.snack_y, self.snack_x, self.objectD["background"]] = 0.0
        
        # ใส่ Snake
        for i, (x, y) in enumerate(self.snake_list):
            if i == len(self.snake_list) - 1: 
                # ตำแหน่งสุดท้ายใน list คือ "หัว"
                obs[y, x, self.objectD["snake_head"]] = 1.0
            else:
                # ที่เหลือคือ "ตัว"
                obs[y, x, self.objectD["snake_body"]] = 1.0
            obs[y, x, self.objectD["background"]] = 0.0

        return obs

    def _render_frame(self):
        import pygame
        self.win.fill(self.colors["black"])
        pygame.draw.rect(self.win, self.colors["red"], [self.snack_x * self.block_size, self.snack_y * self.block_size, self.block_size, self.block_size])
        
        for x, y in self.snake_list:
            pygame.draw.rect(self.win, self.colors["green"], [x * self.block_size, y * self.block_size, self.block_size, self.block_size])
            
        value = self.font.render("Score: " + str(self.score), True, self.colors["white"])
        self.win.blit(value, [0, 0])
        pygame.display.update()

    def close(self):
        if self.render_mode:
            import pygame
            pygame.quit()