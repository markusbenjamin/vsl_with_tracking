from utils.codebase import *

# Function to draw an n x n grid centered at (x, y)
def draw_grid(surface, n, x, y, cell_size, line_color=(0, 0, 0)):
    half_grid_size = (n * cell_size) // 2
    start_x = x - half_grid_size
    start_y = y - half_grid_size
    
    for i in range(n + 1):
        pygame.draw.line(
            surface, line_color, 
            (start_x + i * cell_size, start_y), 
            (start_x + i * cell_size, start_y + n * cell_size)
            )
    for i in range(n + 1):
        pygame.draw.line(
            surface, line_color, 
            (start_x, start_y + i * cell_size), 
            (start_x + n * cell_size, start_y + i * cell_size)
            )

# Pygame setup
pygame.init()

# Set window size
window_size = (600, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Grid Drawing Example")

# Set frame rate
clock = pygame.time.Clock()

# Main loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # Get mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Clear screen with white background
    screen.fill((255, 255, 255))

    # Draw 3x3 grid centered at (300, 300) with cell size of 100
    draw_grid(screen, n=3, x=300, y=300, cell_size=100)

    # Draw a red dot at the mouse position
    pygame.draw.circle(screen, (255, 0, 0), (mouse_x, mouse_y), 5)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate to 60 FPS
    clock.tick(60)
