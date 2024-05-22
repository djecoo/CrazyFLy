import numpy as np
import matplotlib.pyplot as plt


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


def plot_spiral(a, b, theta_max=10 * np.pi, num_points=1000):
    p = archimedean_spiral(a, b, theta_max, num_points)

    plt.figure(figsize=(8, 8))
    plt.plot(p[:, 0], p[:, 1], label=f'Spiral with a={a}, b={b}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Archimedean Spiral')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Example usage
    a = 0
    b = 1
    plot_spiral(a, b)

    # Try different values for a and b
    a = 0
    b = 0.02
    plot_spiral(a, b)
