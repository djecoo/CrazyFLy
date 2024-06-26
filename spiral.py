import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def star(radius=1):
    # Generate theta values from 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, 6)[:-1]
    print(theta)
    order = [0, 2, 4, 1, 3]
    theta = theta[order]
    # Compute the corresponding x and y values
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Add origin at beginning and end
    x = np.concatenate([[0], x, [0]])
    y = np.concatenate([[0], y, [0]])

    points = np.array([x, y]).T
    return points

def circle(radius=1, num_points=100):
    # Generate theta values from 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Compute the corresponding x and y values
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    x = np.concatenate([[0], x, [0]])
    y = np.concatenate([[0], y, [0]])
    points = np.array([x, y]).T
    return points


def archimedean_spiral(a=0, b=0.02, theta_max=10*np.pi, num_points=200):
    # Generate theta values from 0 to theta_max
    theta = np.linspace(0, theta_max, num_points)

    # Compute the corresponding r values
    r = a + b * theta

    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.array([x, y]).T
    return points


def normalize_spiral(spiral_coords, fixed_norm=0.05):

    # Step 1: Compute cumulative distance along the spiral
    distances = np.zeros(spiral_coords.shape[0])
    for i in range(1, len(spiral_coords)):
        distances[i] = distances[i - 1] + np.linalg.norm(spiral_coords[i] - spiral_coords[i - 1])

    # Step 2: Determine the fixed norm (distance between points)
    total_length = distances[-1]

    # Step 3: Generate new distances at fixed intervals
    new_distances = np.linspace(0, total_length, int(total_length / fixed_norm))

    # Step 4: Interpolate the x and y coordinates
    interp_x = interp1d(distances, spiral_coords[:, 0], kind='linear')
    interp_y = interp1d(distances, spiral_coords[:, 1], kind='linear')

    # Get new coordinates
    new_coords = np.vstack((interp_x(new_distances), interp_y(new_distances))).T
    return new_coords


def line():
    x = [0, 0]
    y = [0, -1]
    points = np.array([x, y]).T
    return points


def plot_spiral(a, b, theta_max=10 * np.pi, num_points=1000):
    p = archimedean_spiral(a, b, theta_max, num_points)
    p = normalize_spiral(p)
    print(len(p))

    plt.figure(figsize=(8, 8))
    plt.scatter(p[:, 0], p[:, 1], label=f'Spiral with a={a}, b={b}', s=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Archimedean Spiral')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plus_pattern(edge_length: float = 0.30):
    # Generate the points for the plus pattern
    points = np.array([[0, 0], [0, edge_length], [0, 0], [edge_length, 0], [0, 0], [0, -edge_length], [0, 0], [-edge_length, 0], [0, 0]])
    return points

def squished_spiral(hateur=0.3, inter_space=0.15):
    points = [[0, 0]]
    for j in range(0, 10, 2):
        i = j * inter_space
        k = (j+1) * inter_space
        points.append([hateur, i])
        points.append([0, i])
        points.append([-hateur, i])
        points.append([-hateur, -k])
        points.append([0, -k])
        points.append([hateur, -k])
    return np.array(points)
        
    


if __name__ == '__main__':
    # Try different values for a and b
    points = squished_spiral()
    points = np.array(points)

    # plot
    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1], marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
