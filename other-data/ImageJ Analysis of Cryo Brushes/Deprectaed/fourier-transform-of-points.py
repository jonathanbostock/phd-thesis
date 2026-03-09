### Simulated fourier transform of points
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import scipy.optimize as op
import matplotlib.pyplot as plt
import jbplot

def fibonacci_sphere(samples=1000):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        point_array = np.array([x, y, z])

        r = np.sqrt(np.sum(np.power(point_array, 2)))

        points.append(point_array/r)

    return points

def rotate_sphere_randomly(points):

    matrix = R.random().as_matrix()

    new_points = [np.matmul(matrix, point) for point in points]

    return new_points

def angular_locations(points, band = 0.1):

    theta_values = []

    for point in points:

        if np.abs(point[2]) <= band:

            theta_values.append(np.angle(point[0] + 1j * point[1]))

    return np.array(theta_values) + np.pi

def angular_values(theta_values, spread = 0.25):

    theta_values_degrees = theta_values * 180 / np.pi

    intensity_values = []

    for i in range(360):
        i_float = float(i)
        intensity_values.append(np.sum(np.exp(
            np.multiply(np.power(np.divide(np.subtract(theta_values_degrees, i_float), spread), 2), -0.5))))

    return np.array(intensity_values)

def get_stats(distributions, x_vals = None):

    means = []
    stds = []

    for d in distributions:

        S = 0
        S_x = 0
        S_x2 = 0

        if x_vals == None:
            x_vals = range(len(d))

        for x, intensity in zip(x_vals, d):

            S += intensity
            S_x += x * intensity
            S_x2 += x ** 2 * intensity

        E_x = S_x/S
        E_x2 = S_x2/S

        means.append(E_x)
        stds.append(np.sqrt(E_x2 - E_x**2))

    return means, stds

def sqrt_k(x, k):

    return np.sqrt(x)*k

### Define our constants
band_width = 0.1
simulations = 15
point_increment = 100
tries = 100


points_list = [i*point_increment for i in range(1, simulations+1)]

sphere_list = [rotate_sphere_randomly(fibonacci_sphere(i)) for i in points_list]

maximum_distributions = []

for sphere, points in zip(sphere_list, points_list):
    """
    fig, ax = plt.subplots(1, 1, figsize=[8,8])

    x_list = [point[0] for point in sphere]
    y_list = [point[1] for point in sphere]
    alpha_list = [(2 - point[2])/4 for point in sphere]
    color_list = ["r" if abs(point[2]) < band_width else "b" for point in sphere]

    ax.scatter(x_list, y_list, c=color_list, alpha=alpha_list)
    jbplot.save(fig, "Sphere With {} Points".format(points),
                file_types=["png"])
    """

    # Now run some stuff
    maximum_distribution = np.zeros(180)
    for i in range(tries):
        new_sphere = rotate_sphere_randomly(sphere)
        locs = angular_locations(new_sphere, band=band_width)
        spread = 120/len(locs)
        distribution = angular_values(locs, spread=spread)
        fourier_transform = np.abs(np.fft.fft(distribution)[1:181])
        maximum_distribution[np.argmax(fourier_transform)] += 1

    maximum_distributions.append(maximum_distribution)

maximum_means, maximum_stds = get_stats(maximum_distributions, x_vals = range(1,180))

k, k_sig = op.curve_fit(sqrt_k, points_list, maximum_means)



### Plot everything
fig_1, ax_1 = plt.subplots(1,1, figsize=[6.0, 4.0])

"""
jbplot.plotlineset(ax_1, [range(360)], distribution_list, gradient=True,
                   name_list= ["{} Points".format(i) for i in points_list],
                   gradient_vals = points_list)
ax_1.legend()
jbplot.nice_legend(ax_1)
"""
"""
jbplot.plotlineset(ax, [range(1,180)], fourier_transform_averages, gradient=True,
                   name_list= ["{} Points".format(i) for i in points_list],
                   gradient_vals = points_list)
"""

maximum_array = np.array([np.arange(1,181), *maximum_distributions]).T
maximum_df = pd.DataFrame(data = maximum_array,
                                    columns=["x",*points_list])

jbplot.ridgelinedf(fig_1, maximum_df, y=points_list, gradient=True)

jbplot.save(fig_1, "Fourier transform simulation")

# Figure 2
fig_2, ax_2 = plt.subplots(1, 1, figsize= [4,4])

jbplot.scatter(ax_2, points_list, maximum_means, y_sig_vect = maximum_stds)
jbplot.plotfun(ax_2, lambda x: sqrt_k(x, k),
               x_min = 0, x_max = max(points_list),
               label="y = {:.3g}sqrt(x)".format(k[0]))
jbplot.nice_legend(ax_2)
jbplot.save(fig_2, "Maximum Values of Fourier Transform with Band Width = {}".format(band_width))
