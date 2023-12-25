import numpy as np
from scipy.integrate import dblquad

# Define parameters
x0, y0 = 1.0, 1.0
epsilon = 1e-10  # Small epsilon value to prevent division by zero


# Define the velocity components with epsilon
def traction_field(x, y):
    r1_sq = (x - x0)**2 + (y - y0)**2 + epsilon
    r2_sq = (x + x0)**2 + (y + y0)**2 + epsilon

    v_x = - ((y - y0) / r1_sq) - ((y + y0) / r2_sq)
    v_y = ((x - x0) / r1_sq) + ((x + x0) / r2_sq)

    return v_x, v_y


# Define the integrands for moments
def integrand_xx(x, y):
    v_x, _ = traction_field(x, y)
    return 2 * x * v_x


def integrand_yy(x, y):
    _, v_y = traction_field(x, y)
    return 2 * y * v_y


def integrand_xy(x, y):
    v_x, v_y = traction_field(x, y)
    return x * v_y + y * v_x


def tst__contraction_moments():
    # Perform the integrations
    result_xx, _ = dblquad(integrand_xx, -np.inf, np.inf, lambda y: -np.inf, lambda y: np.inf)
    result_yy, _ = dblquad(integrand_yy, -np.inf, np.inf, lambda y: -np.inf, lambda y: np.inf)
    result_xy, _ = dblquad(integrand_xy, -np.inf, np.inf, lambda y: -np.inf, lambda y: np.inf)

    # Display the results
    print("Moment M_xx:", result_xx)
    print("Moment M_yy:", result_yy)
    print("Moment M_xy:", result_xy)


tst__contraction_moments()
