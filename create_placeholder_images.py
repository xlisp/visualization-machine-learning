import pygame

def create_placeholder_images():
    background = pygame.Surface((288, 512))
    background.fill((135, 206, 235))  # Sky blue
    pygame.image.save(background, "background.png")

    bird = pygame.Surface((34, 24))
    bird.fill((255, 0, 0))  # Red
    pygame.image.save(bird, "bird.png")

    pipe = pygame.Surface((52, 320))
    pipe.fill((0, 255, 0))  # Green
    pygame.image.save(pipe, "pipe.png")

# Call this function before creating the environment
create_placeholder_images()

