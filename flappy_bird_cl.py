import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()

        # Constants
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.BIRD_WIDTH = 34
        self.BIRD_HEIGHT = 24
        self.PIPE_WIDTH = 52
        self.PIPE_GAP = 100
        self.GRAVITY = 0.25
        self.JUMP_VELOCITY = -5

        # Action and observation space
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: jump
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Game state
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.game_over = False

    def reset(self):
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = [self._create_pipe()]
        self.score = 0
        self.game_over = False
        return self._get_observation()

    def step(self, action):
        reward = 0.1  # Small reward for staying alive

        # Bird movement
        if action == 1:
            self.bird_velocity = self.JUMP_VELOCITY
        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity

        # Move pipes
        for pipe in self.pipes:
            pipe[0] -= 2  # Move pipes to the left

        # Remove off-screen pipes and add new ones
        if self.pipes and self.pipes[0][0] + self.PIPE_WIDTH < 0:
            self.pipes.pop(0)
            self.score += 1
            reward = 1  # Reward for passing a pipe

        if len(self.pipes) < 2:
            self.pipes.append(self._create_pipe())

        # Check for collisions
        if self._check_collision():
            self.game_over = True
            reward = -1  # Penalty for dying

        # Check if bird is off screen
        if self.bird_y < 0 or self.bird_y + self.BIRD_HEIGHT > self.SCREEN_HEIGHT:
            self.game_over = True
            reward = -1  # Penalty for going off screen

        observation = self._get_observation()
        info = {"score": self.score}

        return observation, reward, self.game_over, False, info

    def render(self):
        self.screen.fill((135, 206, 250))  # Sky blue background

        # Draw pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe[0], 0, self.PIPE_WIDTH, pipe[1]))  # Upper pipe
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe[0], pipe[1] + self.PIPE_GAP, self.PIPE_WIDTH, self.SCREEN_HEIGHT))  # Lower pipe

        # Draw bird
        pygame.draw.rect(self.screen, (255, 255, 0), (50, int(self.bird_y), self.BIRD_WIDTH, self.BIRD_HEIGHT))

        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

    def _create_pipe(self):
        gap_y = np.random.randint(50, self.SCREEN_HEIGHT - 50 - self.PIPE_GAP)
        return [self.SCREEN_WIDTH, gap_y]

    def _check_collision(self):
        for pipe in self.pipes:
            if (50 < pipe[0] < 50 + self.BIRD_WIDTH and
                (self.bird_y < pipe[1] or self.bird_y + self.BIRD_HEIGHT > pipe[1] + self.PIPE_GAP)):
                return True
        return False

    def _get_observation(self):
        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

# Example usage
if __name__ == "__main__":
    env = FlappyBirdEnv()
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, info = env.step(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    env.close()

