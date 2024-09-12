import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_WIDTH = 57
BIRD_HEIGHT = 41
PILLAR_WIDTH = 86
PILLAR_GAP = 158
PILLAR_SPACING = 324
START_Y = 312
GRAVITY = 0.5
JUMP_VELOCITY = 10
HORIZ_VEL = -4

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        # Action space: jump or not jump
        self.action_space = spaces.Discrete(2)
        # Observation space: bird's y position, y velocity, pillar positions, gaps
        self.observation_space = spaces.Box(low=0, high=SCREEN_HEIGHT, shape=(5,), dtype=np.float32)
        
        # Game state variables
        self.reset()

    def reset(self):
        self.bird_y = START_Y
        self.bird_vel_y = 0
        self.pillars = [self._create_pillar(SCREEN_WIDTH)]
        self.score = 0
        self.done = False
        self.timer = 0
        return self._get_observation()

    def step(self, action):
        # Apply gravity
        self.bird_vel_y += GRAVITY
        self.bird_y += self.bird_vel_y
        
        # Handle jump
        if action == 1:
            self.bird_vel_y = -JUMP_VELOCITY

        # Update pillar positions
        self.pillars = [(x + HORIZ_VEL, gap) for x, gap in self.pillars if x > -PILLAR_WIDTH]
        
        # Add new pillar
        if len(self.pillars) < 3:
            self.pillars.append(self._create_pillar(SCREEN_WIDTH))

        # Check for collisions
        self.done = self._check_collision()

        # Update score
        self.score += 1 if not self.done else 0

        # Get observation
        obs = self._get_observation()

        # Return the step information
        return obs, self.score, self.done, {}

    def _get_observation(self):
        pillar_x, gap_top = self.pillars[0]
        return np.array([self.bird_y, self.bird_vel_y, pillar_x, gap_top, PILLAR_GAP], dtype=np.float32)

    def _create_pillar(self, x):
        gap_top = random.randint(60, SCREEN_HEIGHT - PILLAR_GAP - 60)
        return (x, gap_top)

    def _check_collision(self):
        # Check ground or ceiling
        if self.bird_y < 0 or self.bird_y + BIRD_HEIGHT > SCREEN_HEIGHT:
            return True
        
        # Check pillar collision
        for x, gap_top in self.pillars:
            if SCREEN_WIDTH / 4 < x < SCREEN_WIDTH / 4 + BIRD_WIDTH:
                if not (gap_top < self.bird_y < gap_top + PILLAR_GAP):
                    return True
        
        return False

    def render(self, mode='human'):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Fill background
        screen.fill((135, 206, 250))

        # Draw bird
        pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(SCREEN_WIDTH // 4, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))

        # Draw pillars
        for x, gap_top in self.pillars:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x, 0, PILLAR_WIDTH, gap_top))
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x, gap_top + PILLAR_GAP, PILLAR_WIDTH, SCREEN_HEIGHT - gap_top - PILLAR_GAP))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(str(self.score), True, (255, 255, 255))
        screen.blit(score_text, (SCREEN_WIDTH // 2, 50))

        pygame.display.flip()

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    env = FlappyBirdEnv()
    obs = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # Random action (0 or 1)
        obs, reward, done, _ = env.step(action)
        env.render()

        if done:
            obs = env.reset()
            
    env.close()

