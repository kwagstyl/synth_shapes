import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath


def draw_random_filled_circle(matrix, value=1):
    """
    Draws a filled circle with random center and radius on a given matrix.

    Parameters:
    matrix (np.ndarray): The matrix to draw on.
    value (int or float): The value to fill the circle with.
    """
    max_radius = min(matrix.shape) // 2

    # Generate a random radius ensuring it fits within the matrix
    radius = np.random.randint(2, max_radius )
    
    # Generate random center ensuring the circle fits within the matrix
    x0 = np.random.randint(radius, matrix.shape[0] - radius)
    y0 = np.random.randint(radius, matrix.shape[1] - radius)
    
    y, x = np.ogrid[-x0:matrix.shape[0]-x0, -y0:matrix.shape[1]-y0]
    mask = x*x + y*y <= radius*radius
    matrix[mask] = value
    return matrix


def generate_random_vertices(matrix_shape):
    """
    Generate three random vertices within the matrix ensuring they are not collinear.

    Parameters:
    matrix_shape (tuple): The shape of the matrix.

    Returns:
    np.ndarray: An array of three vertices.
    """
    while True:
        vertices = np.random.randint(0, matrix_shape[0], (3, 2))
        # Calculate the area of the triangle using the determinant method
        area = 0.5 * abs(
            vertices[0, 0] * (vertices[1, 1] - vertices[2, 1]) +
            vertices[1, 0] * (vertices[2, 1] - vertices[0, 1]) +
            vertices[2, 0] * (vertices[0, 1] - vertices[1, 1])
        )
        if area > 1:  # Ensure the area is large enough to avoid nearly collinear points
            return vertices

def draw_random_filled_triangle(matrix, value=1):
    """
    Draws a filled triangle with random vertices on a given matrix.

    Parameters:
    matrix (np.ndarray): The matrix to draw on.
    value (int or float): The value to fill the triangle with.
    """
    vertices = generate_random_vertices(matrix.shape)

    # Create a path object for the triangle
    path = mpath.Path(vertices)
    
    # Create a grid of points
    y, x = np.mgrid[:matrix.shape[0], :matrix.shape[1]]
    points = np.vstack((x.ravel(), y.ravel())).T
    
    # Check which points are inside the triangle
    mask = path.contains_points(points).reshape(matrix.shape)
    
    # Fill the triangle in the matrix
    matrix[mask] = value
    return matrix


def generate_random_square(matrix_shape):
    """
    Generate four vertices of a square within the matrix, ensuring it's fully within the bounds.

    Parameters:
    matrix_shape (tuple): The shape of the matrix.

    Returns:
    np.ndarray: An array of four vertices of the square.
    """
    side_length = np.random.randint(5, min(matrix_shape) // 2)
    half_diagonal = np.sqrt(2 * (side_length / 2) ** 2)

    while True:
        x0 = np.random.uniform(half_diagonal, matrix_shape[0] - half_diagonal)
        y0 = np.random.uniform(half_diagonal, matrix_shape[1] - half_diagonal)

        # Square vertices before rotation
        square = np.array([
            [x0 - side_length / 2, y0 - side_length / 2],
            [x0 + side_length / 2, y0 - side_length / 2],
            [x0 + side_length / 2, y0 + side_length / 2],
            [x0 - side_length / 2, y0 + side_length / 2]
        ])

        # Rotate the square around its center
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        center = np.array([x0, y0])
        rotated_square = np.dot(square - center, rotation_matrix) + center

        if (rotated_square >= 0).all() and (rotated_square < matrix_shape[0]).all():
            return rotated_square

def draw_random_filled_square(matrix, value=1):
    """
    Draws a filled square with random vertices and rotation on a given matrix.

    Parameters:
    matrix (np.ndarray): The matrix to draw on.
    value (int or float): The value to fill the square with.
    """
    vertices = generate_random_square(matrix.shape)

    # Create a path object for the rotated square
    path = mpath.Path(vertices)
    
    # Create a grid of points
    y, x = np.mgrid[:matrix.shape[0], :matrix.shape[1]]
    points = np.vstack((x.ravel(), y.ravel())).T
    
    # Check which points are inside the rotated square
    mask = path.contains_points(points).reshape(matrix.shape)
    
    # Fill the square in the matrix
    matrix[mask] = value
    return matrix



def generate_random_rectangle(matrix_shape):
    """
    Generate four vertices of a rectangle within the matrix, ensuring it's fully within the bounds.

    Parameters:
    matrix_shape (tuple): The shape of the matrix.

    Returns:
    np.ndarray: An array of four vertices of the rectangle.
    """
    while True:
        width = np.random.randint(5, min(matrix_shape) // 2)
        height = np.random.randint(5, min(matrix_shape) // 2)
        if width != height:
            break

    half_diagonal = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

    while True:
        x0 = np.random.uniform(half_diagonal, matrix_shape[0] - half_diagonal)
        y0 = np.random.uniform(half_diagonal, matrix_shape[1] - half_diagonal)

        # Rectangle vertices before rotation
        rectangle = np.array([
            [x0 - width / 2, y0 - height / 2],
            [x0 + width / 2, y0 - height / 2],
            [x0 + width / 2, y0 + height / 2],
            [x0 - width / 2, y0 + height / 2]
        ])

        # Rotate the rectangle around its center
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        center = np.array([x0, y0])
        rotated_rectangle = np.dot(rectangle - center, rotation_matrix) + center

        if (rotated_rectangle >= 0).all() and (rotated_rectangle < matrix_shape[0]).all():
            return rotated_rectangle

def draw_random_filled_rectangle(matrix, value=1):
    """
    Draws a filled rectangle with random vertices and rotation on a given matrix.

    Parameters:
    matrix (np.ndarray): The matrix to draw on.
    value (int or float): The value to fill the rectangle with.
    """
    vertices = generate_random_rectangle(matrix.shape)

    # Create a path object for the rotated rectangle
    path = mpath.Path(vertices)
    
    # Create a grid of points
    y, x = np.mgrid[:matrix.shape[0], :matrix.shape[1]]
    points = np.vstack((x.ravel(), y.ravel())).T
    
    # Check which points are inside the rotated rectangle
    mask = path.contains_points(points).reshape(matrix.shape)
    
    # Fill the rectangle in the matrix
    matrix[mask] = value
    return matrix

def draw_random_filled_ellipse(matrix, value=1):
    """
    Draws a filled ellipse with random parameters on a given matrix,
    ensuring the ellipse is fully within the matrix bounds.

    Parameters:
    matrix (np.ndarray): The matrix to draw on.
    value (int or float): The value to fill the ellipse with.

    Returns:
    np.ndarray: The matrix with the drawn ellipse.
    """
    height, width = matrix.shape
    
    # Set maximum possible semi-axes lengths
    max_radius = min(height, width) // 4

    while True:
        # Generate random parameters for the ellipse
        a = np.random.randint(10, max(11, max_radius))  # semi-major axis
        b = np.random.randint(5, a)  # semi-minor axis
        angle = np.random.uniform(0, 2 * np.pi)

        # Calculate the bounding box of the rotated ellipse
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        bbox_width = 2 * np.sqrt((a * cos_angle)**2 + (b * sin_angle)**2)
        bbox_height = 2 * np.sqrt((a * sin_angle)**2 + (b * cos_angle)**2)

        # Check if the bounding box fits within the matrix
        if bbox_width <= width and bbox_height <= height:
            break

    # Generate center coordinates ensuring the ellipse fits
    center_x = np.random.randint(int(bbox_width/2), width - int(bbox_width/2))
    center_y = np.random.randint(int(bbox_height/2), height - int(bbox_height/2))

    # Create a grid of the matrix indices
    y, x = np.ogrid[:height, :width]

    # Translate the grid to the ellipse center
    x = x - center_x
    y = y - center_y

    # Rotate the coordinates
    x_rot = x * cos_angle + y * sin_angle
    y_rot = -x * sin_angle + y * cos_angle

    # Create the ellipse mask
    ellipse_mask = ((x_rot / a)**2 + (y_rot / b)**2 <= 1)

    # Apply the mask to the matrix
    matrix[ellipse_mask] = value

    return matrix



def draw_random_shape(matrix, value=1, shape='random'):
    """
    Draws a random shape on a matrix.

    Parameters:
    matrix_size (tuple): The size of the matrix (height, width).
    value (int or float): The value to fill the shape with.
    shape (str): The shape to draw. Options are 'circle', 'triangle', 'square', 'rectangle', 'ellipse', or 'random'.

    Returns:
    np.ndarray: The matrix with the drawn shape.
    """
    
    shape_functions = {
        'circle': draw_random_filled_circle,
        'triangle': draw_random_filled_triangle,
        'square': draw_random_filled_square,
        'rectangle': draw_random_filled_rectangle,
        'ellipse': draw_random_filled_ellipse
    }
    
    if shape == 'random':
        shape = np.random.choice(list(shape_functions.keys()))
    
    if shape not in shape_functions:
        raise ValueError(f"Invalid shape: {shape}. Choose from {', '.join(shape_functions.keys())} or 'random'.")
    
    return shape_functions[shape](matrix, value)
