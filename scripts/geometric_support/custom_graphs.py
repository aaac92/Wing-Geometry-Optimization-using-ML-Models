import matplotlib.pyplot as plt
import numpy as np
from .custom_filters import compute_true_thickness

def plot_LE_slopes(x, zu, zl, n_points=10):
    """
    Plot slope vectors and angles at the LE for upper and lower surfaces.

    Parameters:
        x, zu, zl : arrays of x and z coordinates
        n_points  : number of points near LE to plot
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    x_u, z_u = x[:n_points], zu[:n_points]
    x_l, z_l = x[:n_points], zl[:n_points]

    ax.plot(x, zu, label="Upper Surface", color='blue')
    ax.plot(x, zl, label="Lower Surface", color='green')
    ax.scatter(x[:n_points], zu[:n_points], color='blue')
    ax.scatter(x[:n_points], zl[:n_points], color='green')

    # First segments
    def plot_vector(x_start, z_start, vec, color, label=None):
        ax.quiver(x_start, z_start, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                  color=color, width=0.005, label=label)

    # Upper and lower vectors
    vec_u = np.array([x_u[1] - x_u[0], z_u[1] - z_u[0]])
    vec_l = np.array([x_l[1] - x_l[0], z_l[1] - z_l[0]])

    # Plot slope vectors
    plot_vector(x_u[0], z_u[0], vec_u, 'red', 'Upper LE Vector')
    plot_vector(x_l[0], z_l[0], vec_l, 'orange', 'Lower LE Vector')

    # Plot vertical line at LE
    #ax.quiver(0, (zu[0] + zl[0]) / 2, 0, 0.1, angles='xy', scale_units='xy', scale=1,
    #      color='black', label='Vertical')
    ax.plot([0, 0], [min(zl[:n_points]), max(zu[:n_points])], 'k--', label='Vertical Ref')

    ax.set_title("LE Slope Vectors and Vertical Reference")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend()
    ax.grid(True)
    plt.axis('equal')
    plt.show()

def plot_thickness_behavior(x, zu, zl, fail_x=None):
    thickness = compute_true_thickness(x, zu, zl)
    zc = 0.5 * (zu + zl)

    plt.figure(figsize=(15, 6))

    # Plot upper and lower surfaces
    plt.plot(x, zu, label='Upper Surface', color='black')
    plt.plot(x, zl, label='Lower Surface', color='black')

    # Plot camber line and thickness curve
    plt.plot(x, zc, label='Camber Line', color='gray', linestyle=':')
    plt.plot(x, thickness, label='True Thickness', color='blue', linestyle='--')

    if fail_x is not None:
        # Highlight failure point
        fail_thickness = np.interp(fail_x, x, thickness)
        plt.axvline(fail_x, color='red', linestyle='--', label=f'Non-monotonicity at x={fail_x:.3f}')
        plt.scatter(fail_x, fail_thickness, color='red', zorder=5)

    else:
        plt.legend(["Upper Surface", "Lower Surface", "Camber Line", "Perpendicular Thickness", "Airfoil thickness is monotonic"])

    plt.xlabel('x')
    plt.ylabel('z / Thickness')
    plt.axis('equal')
    plt.title('Airfoil Geometry and True Thickness Distribution')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()