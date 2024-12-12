"""
Function repository and dependency imports.
"""

#region Imports
import numpy as np
from numpy import typing as npt
import os
import traceback
import pygame
import sys
import scipy.ndimage
import math
from screeninfo import get_monitors
#endregion

#region Misc utils
def get_monitor_ppcs():
    """
    Returns the pixel density per cm for all the monitors.
    """
    ppcs = []
    for monitor in get_monitors():
        ppc_x = monitor.width / (monitor.width_mm / 25.4) / 2.54
        ppc_y = monitor.height / (monitor.height_mm / 25.4) / 2.54
        ppcs.append((ppc_x + ppc_y) / 2)  # Average pixels per cms
    return ppcs

def pixels_to_visual_angles(pixels, ppc, viewing_dist):
    """
    Convert pixels to visual angle in radians based on pixel density and viewing distance.

    Parameters:
    - pixels (float): Number of pixels to convert.
    - ppc (float): Pixel density in pixels per centimeter.
    - viewing_dist (float): Viewing distance in centimeters.

    Returns:
    - float: Visual angle in radians.
    """
    return 2 * np.arctan((pixels / ppc) / (2 * viewing_dist))

def visual_angles_to_pixels(visual_angle_rad, ppc, viewing_dist):
    """
    Convert visual angle in radians to pixels based on pixel density and viewing distance.

    Parameters:
    - visual_angle_rad (float): Visual angle in radians.
    - ppc (float): Pixel density in pixels per centimeter.
    - viewing_dist (float): Viewing distance in centimeters.

    Returns:
    - float: Number of pixels corresponding to the given visual angle.
    """
    return 2 * viewing_dist * np.tan(visual_angle_rad / 2) * ppc

def round_to(number, multiple):
    """
    Rounds a number to the nearest multiple of a specified value.

    :param number: The number to round.
    :param multiple: The value to round to the nearest multiple of.
    :return: The rounded number.
    """
    return round(number / multiple) * multiple
#endregion

#region Movement dynamics
#region Target
class Target:
    """
    Target with gaussian random walk on position and velocity, rectangular boundary conditions, dampening and force. 
    Adapted from Dominik Straub's RandomWalker class.
    
    $$
    \\begin{align*}
    
    pos &= \\alpha \\cdot pos + dt \\cdot vel + \\sqrt{dt} \\cdot \\varepsilon  & \\varepsilon \\sim \\mathcal{N}(0, \\sigma_{pos}) \\
    vel &= \\beta \\cdot vel + \\sqrt{dt} \\cdot \\eta & \\eta \\sim \\mathcal{N}(0, \\sigma_{vel})
    
    \\end{align*}
    $$
    """

    def __init__(
        self,
        init_pos: npt.ArrayLike,
        std_pos: float,
        std_vel: float = 0.0,
        dt: float = 1 / 60,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.rng = np.random.default_rng()

        self.init_pos = np.array(init_pos, ndmin=1)
        self.dim = len(self.init_pos)

        self.std_pos = std_pos
        self.std_vel = std_vel
        self.dt = dt
        self.alpha = alpha
        self.beta = beta

        self.pos = self.init_pos
        self.vel = np.zeros(self.dim)

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.init_pos=}, {self.std_pos=}, {self.std_vel=}, "
                f"{self.dt=}, {self.alpha=}, {self.beta=})")
    
    def apply_hard_boundaries(self, x_min, x_max, y_min, y_max) -> None:
        """
        Implements a non-physical clipping of the position vector.
        """
        x, y = self.pos

        x = np.clip(x, x_min + 1, x_max - 1)
        y = np.clip(y, y_min + 1, y_max - 1)

        self.pos = np.array([x, y])

    def apply_reflective_boundaries(self, x_min, x_max, y_min, y_max) -> None:
        """
        Implements a fully elastic bounce.
        """
        x, y = self.pos
        vx, vy = self.vel

        if x < x_min:
            x = x_min + (x_min - x)
            vx = -vx
        elif x > x_max:
            x = x_max - (x - x_max)
            vx = -vx

        if y < y_min:
            y = y_min + (y_min - y)
            vy = -vy
        elif y > y_max:
            y = y_max - (y - y_max)
            vy = -vy

        self.pos = np.array([x, y])
        self.vel = np.array([vx, vy])
    
    def apply_forces(self, force) -> None:
        """
        Update velocity based on net force vector.
        """
        self.vel = self.vel + self.dt * np.array(force)
    
    def dissipate_kinetic_energy(self, dampening) -> None:
        """
        Dampens the velocity.
        """
        self.vel *= dampening

    def walk(self) -> None:
        """
        Performs one step, i.e. calculates the next position.
        """
        pos_next = self.alpha * self.pos + self.dt * self.vel + np.sqrt(self.dt) * self.rng.normal(scale=self.std_pos, size=self.dim)
        vel_next = self.beta * self.vel + np.sqrt(self.dt) * self.rng.normal(scale=self.std_vel, size=self.dim)

        self.pos = pos_next
        self.vel = vel_next

    def get_vel(self) -> tuple:
        return self.vel

    def get_pos(self) -> tuple:
        return self.pos

    def set_pos(self, pos:tuple):
        self.pos = pos
    
    def reset_pos(self):
        self.pos = self.init_pos
#endregion

#region Repulsive potential
def compute_force_due_to_repulsive_potential(
    pos,
    x_min,
    x_max,
    y_min,
    y_max,
    k,
    alpha
):
    """
    Compute the force due to a potential function at a given position (x, y).
    
    Parameters:
        pos (array-like): Position vector [x, y].
        x_min (float): Minimum x-coordinate of the rectangle. Default is 0.0.
        x_max (float): Maximum x-coordinate of the rectangle. Default is 10.0.
        y_min (float): Minimum y-coordinate of the rectangle. Default is 0.0.
        y_max (float): Maximum y-coordinate of the rectangle. Default is 5.0.
        k (float): Potential strength constant. Default is 1.0.
        alpha (float): Controls the steepness of the potential. Default is 2.0.
    
    Returns:
        np.ndarray: Force vector [F_x, F_y] due to the potential at the given position.
    """
    x, y = pos
    epsilon = 1e-10  # Small value to prevent division by zero

    if x == x_min:
        x += 1
    elif x == x_max:
        x -= 1
    if y == y_min:
        y += 1
    elif y == y_max:
        y -= 1

    # Compute partial derivatives (forces) in x and y directions
    Fx = k * alpha * (
        1 / ((x - x_min + epsilon) ** (alpha + 1))
        - 1 / ((x_max - x + epsilon) ** (alpha + 1))
    )
    
    Fy = k * alpha * (
        1 / ((y - y_min + epsilon) ** (alpha + 1))
        - 1 / ((y_max - y + epsilon) ** (alpha + 1))
    )

    return (Fx, Fy)
#endregion
#endregion

#region Draw
def draw_text(screen, text, font, offset, text_color=(255, 255, 255)):
    """
    Draws centered white text in a bounding box on the given Pygame screen.

    Parameters:
    - screen (pygame.Surface): The Pygame surface to draw on.
    - text (str): The text to display.
    - font (pygame.font.Font): The font to use for rendering text.
    - offset: From the center in pixels.
    - text_color (tuple): RGB color of the text (default white).
    """
    # Render the text surface
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(screen.get_rect().centerx + offset[0], screen.get_rect().centery + offset[1]))  # Center the text

    # Draw the box and then blit the text
    screen.blit(text_surface, text_rect)

def draw_mouse(screen, mouse_pos):
    pygame.draw.circle(screen, (125, 0, 0), (mouse_pos[0], mouse_pos[1]), 10)

def draw_grid(surface, n, x, y, cell_size, line_width = 3, line_color=(0, 0, 0)):
    half_grid_size = (n * cell_size) // 2
    start_x = x - half_grid_size
    start_y = y - half_grid_size
    
    for i in range(n + 1):
        pygame.draw.line(
            surface, line_color, 
            (start_x + i * cell_size, start_y), 
            (start_x + i * cell_size, start_y + n * cell_size),
            width = line_width
            )
    for i in range(n + 1):
        pygame.draw.line(
            surface, line_color, 
            (start_x, start_y + i * cell_size), 
            (start_x + n * cell_size, start_y + i * cell_size),
            width = line_width
            )
        
def draw_shapes(screen, x, y, cell_size, shapes_data, shapes_to_draw, positions):
    for i, pos in enumerate(positions):
        shape = shapes_data[shapes_to_draw[i]]  # shapes_to_draw is not zero-indexed
        grid_x = x + (pos[1] - 2) * cell_size - cell_size/2  # Column
        grid_y = y + (pos[0] - 2) * cell_size - cell_size/2 # Row
        screen.blit(shape, (grid_x, grid_y))

def generate_blended_noise_mask(width, height, noise_strength=255, noise_density=0.5, smoothing=1, blend_alpha=128, scale_factor = 1):
    """
    Generates a smooth, semi-transparent noise mask at position (x, y) with specified width, height, and noise strength.
    Uses a downscaled noise mask and then upscales it to improve performance.
    
    Parameters:
    - screen: Pygame surface to draw the noise on
    - x, y: Center coordinates of the noise mask
    - width, height: Dimensions of the noise mask
    - noise_strength: Intensity of noise (0 to 255)
    - noise_density: Proportion of noisy pixels (0 to 1)
    - smoothing: Degree of Gaussian smoothing to apply (higher values = smoother noise)
    - blend_alpha: Alpha transparency for blending (0 = fully transparent, 255 = fully opaque)
    - scale_factor: Factor by which the noise is downscaled (e.g., 0.5 means half size)
    """

    # Calculate the dimensions of the smaller noise mask
    small_width = int(width * scale_factor)
    small_height = int(height * scale_factor)

    # Generate random noise in the smaller resolution
    small_noise_array = np.random.rand(small_height, small_width) * noise_strength * noise_density

    # Apply Gaussian smoothing
    if 0<smoothing:
        small_noise_array = scipy.ndimage.gaussian_filter(small_noise_array, smoothing)

    # Normalize noise to 0-255 range
    small_noise_array = np.clip(small_noise_array, 0, 255).astype(np.uint8)

    # Create an RGBA array by adding an alpha channel for blending
    small_noise_surface_array = np.stack([small_noise_array] * 3 + [np.full_like(small_noise_array, blend_alpha)], axis=-1)

    # Create a Pygame surface and use surfarray to convert numpy array into a surface
    small_noise_surface = pygame.surfarray.make_surface(small_noise_surface_array[:, :, :3])  # RGB only
    small_noise_surface.set_alpha(blend_alpha)  # Apply alpha transparency

    # Upscale the smaller noise surface to the full size
    return pygame.transform.smoothscale(small_noise_surface, (width, height))

def draw_blended_noise_mask(screen, x, y, mask):
    noise_surface = mask

    # Calculate top-left corner to center the noise mask
    top_left_x = x - mask.get_width() // 2
    top_left_y = y - mask.get_height() // 2

    # Draw noise surface onto the screen with alpha blending
    screen.blit(noise_surface, (top_left_x, top_left_y))

def generate_gaussian_blob(size, min_size=0.1):
    """
    Creates a Gaussian blob in Pygame. The 'size' parameter corresponds to the standard deviation of the Gaussian.
    The 'min_size' is the minimum bounding box size for the blob. The blob is drawn as an RGBA surface with transparency.
    
    Parameters:
    - size: Standard deviation of the Gaussian
    - min_size: Minimum size of the blob (default 0.1)

    Adapted from Dominik Straub.
    """
    
    # ±3 standard deviations contain 99.7% of the mass
    s = size * 6
    min_s = min_size * 6

    if s == 0:
        # Create a "perfectly" visible, black circle for a minimal blob
        radius = min_s / 4
        blob_surface = pygame.Surface((min_s, min_s), pygame.SRCALPHA)
        pygame.draw.circle(blob_surface, (0, 0, 0, 255), (min_s // 2, min_s // 2), int(radius))
        return blob_surface

    if s == math.inf:
        # Invisible blob (return empty surface)
        return pygame.Surface((0, 0), pygame.SRCALPHA)

    # Create a Gaussian array with numpy
    blob_size = int(s)
    x = np.linspace(-3, 3, blob_size)
    y = np.linspace(-3, 3, blob_size)
    x, y = np.meshgrid(x, y)
    gauss = np.exp(-(x**2 + y**2) / 2)  # Standard Gaussian formula

    # Normalize the Gaussian to 0-255 range for transparency
    gauss = np.clip(gauss * 255, 0, 255).astype(np.uint8)

    # Create an RGBA surface for the Gaussian blob
    blob_surface = pygame.Surface((blob_size, blob_size), pygame.SRCALPHA)

    # Set the Gaussian blob's alpha channel
    for i in range(blob_size):
        for j in range(blob_size):
            alpha = gauss[i, j]
            blob_surface.set_at((j, i), (255, 255, 255, alpha))  # White blob with Gaussian alpha

    return blob_surface

def draw_gaussian_blob(screen,blob,x,y):
    screen.blit(blob, (x - blob.get_width() // 2, y - blob.get_height() // 2))
#endregion

#region Logging
def set_up_logging(session_directory,filename):
    file_path = os.path.join(session_directory, filename+'.txt')
    with open(file_path, 'w') as file: # Create it
        pass
    return file_path

def log(message, file_path, print_to_console = True):
    if print_to_console:
        print(message)
    with open(file_path, 'a') as file:
        print(message, file=file)

#endregion

#region Trial types

def familiarization_trial(scenes, scene,  shape_mapping, word_decode):
    word1 = scenes[scene-1, 3-1]
    word2 = scenes[scene-1, 6-1]
    word3 = scenes[scene-1, 9-1]

    shape1 = word_decode[word1-1, 1-1]
    shape2 = word_decode[word1-1, 2-1]
    shape3 = word_decode[word2-1, 1-1]
    shape4 = word_decode[word2-1, 2-1]
    shape5 = word_decode[word3-1, 1-1]
    shape6 = word_decode[word3-1, 2-1]

    shape1 = shape_mapping[shape1-1]
    shape2 = shape_mapping[shape2-1]
    shape3 = shape_mapping[shape3-1]
    shape4 = shape_mapping[shape4-1]
    shape5 = shape_mapping[shape5-1]
    shape6 = shape_mapping[shape6-1]

    shapes = [shape1, shape2, shape3, shape4, shape5, shape6]

    row1 = scenes[scene-1, 1-1]
    col1 = scenes[scene-1, 2-1]
    row3 = scenes[scene-1, 4-1]
    col3 = scenes[scene-1, 5-1]
    row5 = scenes[scene-1, 7-1]
    col5 = scenes[scene-1, 8-1]

    row2 = row1 + word_decode[word1-1, 3-1]
    col2 = col1 + word_decode[word1-1, 4-1]
    row4 = row3 + word_decode[word2-1, 3-1]
    col4 = col3 + word_decode[word2-1, 4-1]
    row6 = row5 + word_decode[word3-1, 3-1]
    col6 = col5 + word_decode[word3-1, 4-1]

    pos1 = [col1,row1]
    pos2 = [col2,row2]
    pos3 = [col3,row3]
    pos4 = [col4,row4]
    pos5 = [col5,row5]
    pos6 = [col6,row6]

    positions = [pos1, pos2, pos3, pos4, pos5, pos6]

    return {
        'shapes': shapes,
        'positions': positions
    }

def pairs_test_trial(scenes, scene, shape_mapping, word_decode, nonword_decode, randomized_first_pres):
    word = scenes[scene-1, 3-1]
    nonword = scenes[scene-1, 6-1]

    shape1 = word_decode[word-1, 1-1]
    shape2 = word_decode[word-1, 2-1]
    shape3 = nonword_decode[nonword-1, 1-1]
    shape4 = nonword_decode[nonword-1, 2-1]

    shape1 = shape_mapping[shape1-1]
    shape2 = shape_mapping[shape2-1]
    shape3 = shape_mapping[shape3-1]
    shape4 = shape_mapping[shape4-1]

    shapes = [
        [shape1, shape2],
        [shape3, shape4]
    ]

    row1 = scenes[scene-1, 1-1]
    col1 = scenes[scene-1, 2-1]
    row3 = scenes[scene-1, 4-1]
    col3 = scenes[scene-1, 5-1]

    row2 = row1 + word_decode[word-1, 3-1]
    col2 = col1 + word_decode[word-1, 4-1]
    row4 = row3 + nonword_decode[nonword-1, 3-1]
    col4 = col3 + nonword_decode[nonword-1, 4-1]

    positions = [
        [
            [col1, row1],
            [col2, row2]
        ],
        [
            [col3, row3],
            [col4, row4]
        ]
    ]

    if randomized_first_pres:
        first_pres = 1 + np.random.randint(2)
    else:
        first_pres = scenes[scene-1, 7-1]

    return {
        'shapes': shapes,
        'positions': positions,
        'first_pres': first_pres
    }

def singles_test_trial(scenes, scene, shape_mapping):
    shape1 = scenes[scene-1, 3-1]  # frequent
    shape2 = scenes[scene-1, 6-1]  # infrequent

    shape1 = shape_mapping[shape1-1]
    shape2 = shape_mapping[shape2-1]

    shapes = [[shape1], [shape2]]

    row1 = scenes[scene-1, 1-1]
    col1 = scenes[scene-1, 2-1]
    row2 = scenes[scene-1, 4-1]
    col2 = scenes[scene-1, 5-1]

    positions = [
        [[col1, row1]],
        [[col2, row2]]
    ]

    first_pres = scenes[scene-1, 7-1]

    return {
        'shapes': shapes,
        'positions': positions,
        'first_pres': first_pres
    }
#endregion