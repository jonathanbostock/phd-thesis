### Simulated fourier transform of points
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import scipy.signal as signal
import scipy.optimize as op
import scipy.stats as stats
import matplotlib.pyplot as plt
import jbplot
from PIL import Image, ImageDraw, ImageFilter
import cv2


def fibonacci_sphere(samples=1000, noise = 0):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        x += np.random.normal(0, noise)
        y += np.random.normal(0, noise)
        z += np.random.normal(0, noise)

        point_array = np.array([x, y, z])

        r = np.sqrt(np.sum(np.power(point_array, 2)))

        points.append(point_array/r)

    return np.array(points)

def rotate_sphere_randomly(points):

    matrix = R.random().as_matrix()

    new_points = [np.matmul(matrix, point) for point in points]

    return np.array(new_points)

def analyse_sphere(points, r_factor = 0.6,
                   dim = 4096, save=False):

    image = np.zeros((dim, dim, 3), np.uint8)

    start_points = np.add(np.multiply(points[...,:-1], dim/4), dim/2)
    end_points = np.add(np.multiply(points[...,:-1], dim/4*(1+r_factor)), dim/2)
    lines = np.int32(np.transpose(np.array([start_points, end_points]), axes=(1, 0, 2)))

    for line in lines:
        cv2.polylines(image, np.array([line], dtype=np.int32), 1, (255, 255, 255), 20)

    image_blurred = cv2.filter2D(image, 3, 3, 5).astype(np.uint8)

    polar_transform = np.array(cv2.linearPolar(image_blurred,(dim/2, dim/2),
                                               int((1+r_factor)*dim/4),
                                               cv2.WARP_FILL_OUTLIERS))[..., 0]

    polar_image = Image.fromarray(polar_transform)

    if save:
        Image.fromarray(image_blurred).convert("L").save("Image.png")
        polar_image.convert("L").save("Image 2.png")

    radial_fft = np.fft.fft(polar_transform, axis=0)

    fft_average_start = int(dim/2)
    fft_average_end = int(dim/2*(1+r_factor))
    fft_average_c = np.average(radial_fft[1:180,fft_average_start:fft_average_end], axis=1)
    fft_average = np.abs(fft_average_c)

    return np.argmax(fft_average) + 1

point_numbers = [100*i for i in range(1,11)]
samples = 100

frequency_array = [np.array(range(1,180))]
means = []
stdevs = []
for n in point_numbers:
    base_sphere = fibonacci_sphere(samples=n)

    values = [analyse_sphere(rotate_sphere_randomly(base_sphere), save=False)
              for i in range(samples)]

    means.append(np.mean(values))
    stdevs.append(stats.tstd(values))

    frequencies = np.zeros([179])

    for v in values:
        frequencies[v-1] += 1

    frequency_array.append(frequencies)

frequency_array = np.array(frequency_array)
frequency_df = pd.DataFrame(frequency_array.transpose(),columns =[ "x", *point_numbers])



### Plot
fig_1, ax_1 = plt.subplots(1, 1, figsize=[4, 6])

jbplot.ridgelinedf(fig_1, frequency_df, y = point_numbers, gradient=True)

jbplot.save(fig_1, "Simulations 2", file_types = ["png"])


fig_2, ax_2 = plt.subplots(1, 1, figsize=[4,4])
jbplot.scatter(ax_2, point_numbers, means, y_sig_vect = stdevs)

jbplot.save(fig_2, "Max Values of Simulations 2", file_types = ["png"])
