import pygame
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import random

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.BIRD_WIDTH = 34
        self.BIRD_HEIGHT = 24
        self.PIPE_WIDTH = 52
        self.PIPE_GAP = 100
        self.GRAVITY = 1
        self.JUMP_STRENGTH = 10
        self.PIPE_VELOCITY = 3

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Load images
        self.background_img = pygame.image.load('background.png').convert()
        self.bird_img = pygame.image.load('bird.png').convert_alpha()
        self.pipe_img = pygame.image.load('pipe.png').convert_alpha()

        self.reset()

    def get_state(self):
        next_pipe = self.pipes[0]
        return np.array([
            self.bird_y,
            self.bird_velocity,
            next_pipe[0] - 50,  # x distance to next pipe
            next_pipe[1] - self.bird_y,  # y distance to next pipe's top
            next_pipe[1] + self.PIPE_GAP - self.bird_y  # y distance to next pipe's bottom
        ])

    def check_collision(self):
        """Check for collisions with pipes or the ground."""
        # Bird rectangle for collision detection
        bird_rect = pygame.Rect(50, self.bird_y, self.BIRD_WIDTH, self.BIRD_HEIGHT)

        # Check if the bird hits the ground or flies off-screen upwards
        if self.bird_y + self.BIRD_HEIGHT >= self.SCREEN_HEIGHT or self.bird_y < 0:
            return True

        # Check if the bird collides with pipes
        for pipe in self.pipes:
            # Top and bottom pipe rects
            top_pipe_rect = pygame.Rect(pipe[0], pipe[1] - self.PIPE_GAP - 320, self.PIPE_WIDTH, 320)
            bottom_pipe_rect = pygame.Rect(pipe[0], pipe[1], self.PIPE_WIDTH, self.SCREEN_HEIGHT - pipe[1])

            # Check collision with the top or bottom pipe
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                return True

        return False

    def step(self, action):
        reward = 0
        self.bird_velocity += self.GRAVITY
        if action == 1:
            self.bird_velocity = -self.JUMP_STRENGTH
        self.bird_y += self.bird_velocity

        # Move pipes
        for pipe in self.pipes:
            pipe[0] -= self.PIPE_VELOCITY

        # Remove passed pipes and add new ones
        if self.pipes[0][0] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
            self.score += 1
            reward = 1
            new_pipe = [self.SCREEN_WIDTH, random.randint(100, self.SCREEN_HEIGHT - 100 - self.PIPE_GAP)]
            self.pipes.append(new_pipe)

        # Check for collisions
        if self.check_collision():
            reward = -1
            return self.get_state(), reward, True, False, {}

        return self.get_state(), reward, False, False, {}

    def reset(self):
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = [[self.SCREEN_WIDTH, random.randint(100, self.SCREEN_HEIGHT - 100 - self.PIPE_GAP)]]
        self.score = 0
        return self.get_state()

    def render(self):
        self.screen.blit(self.background_img, (0, 0))

        # Draw pipes
        for pipe in self.pipes:
            self.screen.blit(self.pipe_img, (pipe[0], pipe[1] - self.PIPE_GAP - 320))
            self.screen.blit(pygame.transform.flip(self.pipe_img, False, True), (pipe[0], pipe[1] + self.PIPE_GAP))

        # Draw bird
        self.screen.blit(self.bird_img, (50, self.bird_y))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

####### -------- main ---------
# Example usage with player control
if __name__ == "__main__":
    env = FlappyBirdEnv()
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = 0  # Default action is do nothing

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1  # Jump when spacebar is pressed
            if event.type == pygame.MOUSEBUTTONDOWN:
                action = 1  # Jump when mouse is clicked

        obs, reward, done, _, info = env.step(action)

        if done:
            print(f"Game Over! Final Score: {env.score}")
            # Wait for a moment before closing
            pygame.time.wait(2000)

    env.close()

