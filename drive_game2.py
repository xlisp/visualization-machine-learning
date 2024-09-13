import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import random
import os

class DrivingEnv(gym.Env):
    def __init__(self):
        super(DrivingEnv, self).__init__()
        self.width = 800
        self.height = 600
        self.car_width = 60
        self.car_height = 100
        self.obstacle_width = 60
        self.obstacle_height = 100
        self.car_speed = 5
        self.obstacle_speed = 3
        self.lanes = [self.width // 4, self.width // 2, 3 * self.width // 4]

        self.action_space = spaces.Discrete(3)  # Left, Stay, Right
        self.observation_space = spaces.Box(low=0, high=255, shape=(5,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.car_pos = [self.width // 2, self.height - self.car_height - 10]
        self.obstacles = []
        self.score = 0

        # Load images
        self.player_car_img = self.load_image("player_car.png", (self.car_width, self.car_height))
        self.obstacle_car_img = self.load_image("obstacle_car.png", (self.obstacle_width, self.obstacle_height))

    def load_image(self, filename, size):
        try:
            image = pygame.image.load(os.path.join("assets", filename))
            return pygame.transform.scale(image, size)
        except pygame.error as e:
            print(f"Unable to load image {filename}: {e}")
            # Create a colored rectangle as a fallback
            surface = pygame.Surface(size)
            surface.fill((255, 0, 0) if "player" in filename else (0, 0, 255))
            return surface

    def reset(self):
        self.car_pos = [self.width // 2, self.height - self.car_height - 10]
        self.obstacles = []
        self.score = 0
        return self.get_state()

    def step(self, action):
        # Move the car
        if action == 0 and self.car_pos[0] > self.lanes[0]:  # Left
            self.car_pos[0] -= self.car_speed
        elif action == 2 and self.car_pos[0] < self.lanes[2]:  # Right
            self.car_pos[0] += self.car_speed

        # Move obstacles
        for obstacle in self.obstacles:
            obstacle[1] += self.obstacle_speed

        # Remove obstacles that are off-screen
        self.obstacles = [obs for obs in self.obstacles if obs[1] < self.height]

        # Add new obstacles
        if random.random() < 0.02:
            new_obstacle = [random.choice(self.lanes), -self.obstacle_height]
            self.obstacles.append(new_obstacle)

        # Check for collisions
        car_rect = pygame.Rect(self.car_pos[0] - self.car_width // 2, self.car_pos[1], self.car_width, self.car_height)
        for obstacle in self.obstacles:
            obs_rect = pygame.Rect(obstacle[0] - self.obstacle_width // 2, obstacle[1], self.obstacle_width, self.obstacle_height)
            if car_rect.colliderect(obs_rect):
                return self.get_state(), -10, True, False, {}

        # Update score
        self.score += len([obs for obs in self.obstacles if obs[1] > self.car_pos[1] + self.car_height])

        return self.get_state(), 1, False, False, {}

    def get_state(self):
        if not self.obstacles:
            return [self.car_pos[0] / self.width, 1, 1, 1, 0]
        
        closest_obstacles = sorted(self.obstacles, key=lambda obs: obs[1])[:3]
        while len(closest_obstacles) < 3:
            closest_obstacles.append([self.width // 2, -self.obstacle_height])

        state = [
            self.car_pos[0] / self.width,
            closest_obstacles[0][0] / self.width,
            closest_obstacles[0][1] / self.height,
            closest_obstacles[1][0] / self.width,
            closest_obstacles[1][1] / self.height,
        ]
        return state

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw lanes
        for lane in self.lanes:
            pygame.draw.line(self.screen, (200, 200, 200), (lane, 0), (lane, self.height), 2)

        # Draw player car
        self.screen.blit(self.player_car_img, (self.car_pos[0] - self.car_width // 2, self.car_pos[1]))

        # Draw obstacles
        for obstacle in self.obstacles:
            self.screen.blit(self.obstacle_car_img, (obstacle[0] - self.obstacle_width // 2, obstacle[1]))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def human_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return 0  # Left
                elif event.key == pygame.K_RIGHT:
                    return 2  # Right
        return 1  # Stay (default action)

    def close(self):
        if self.screen is not None:
            pygame.quit()

def play_human():
    env = DrivingEnv()
    env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        env.render()
        action = env.human_input()
        
        if action is None:  # Quit the game
            break
        
        state, reward, done, truncated, info = env.step(action)
        print(f"Score: {env.score}, Reward: {reward}")
        
        clock.tick(30)  # Limit to 30 FPS

    env.close()
    print(f"Game Over! Final Score: {env.score}")

if __name__ == "__main__":
    play_human()

