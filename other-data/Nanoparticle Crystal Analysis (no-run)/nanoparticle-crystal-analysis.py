# Jonathan Bostock 07-Sep-2023
# Nanoparticle Crystallization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# import UpdateParticlesImport

def main():

    particle_number:    int     = 50
    dimensions:         int     = 2
    standard_deviation: float   = 0.2
    charges:            bool    = True
    iterations:         int     = 3000

    particle_list = create_particles(particle_number, dimensions, standard_deviation, charges)
    particle_list = anneal(particle_list, iterations)
    particle_list = clean(particle_list)
    plot_particles(particle_list, data_string(particle_number,
                                              dimensions,
                                              standard_deviation,
                                              charges,
                                              iterations),
                   dimensions)

# Deprecated but kept around, shouldn't be used
def morse_force(particle_1: list, particle_2: list) -> list:

    p_1, r_1, c_1 = particle_1
    p_2, r_2, c_2 = particle_2

    displacement = np.subtract(p_2, p_1)
    distance = np.linalg.norm(displacement)
    distance_anomaly = np.subtract(distance, np.add(r_1, r_2))
    displacement_normalized = np.divide(displacement, distance)

    exp_term = np.exp(-distance_anomaly)
    force_magnitude = np.product([2, exp_term, np.subtract(1, exp_term)])

    return np.multiply(force_magnitude, displacement_normalized)

def lj_force(particle_1: list, particle_2: list) -> list:

    p_1, r_1, c_1 = particle_1
    p_2, r_2, c_2 = particle_2

    displacement = np.subtract(p_2, p_1)
    distance = np.linalg.norm(displacement)
    distance_anomaly = np.divide(distance, np.add(r_1, r_2))
    displacement_normalized = np.divide(displacement, distance)

    # This bit is kinda fucked, but it lets you get a coefficient quickly with indexing
    # If both charges are zero, attraction occurs
    # If they are 1 and -1, attraction occurs
    # Else no attraction
    attraction_coefficient = ((1.0, 0.0, 0.0),
                              (0.0, 0.0, 1.0),
                              (0.0, 1.0, 0.0))[c_1][c_2]

    force_magnitude = np.subtract(np.multiply(np.power(distance_anomaly, -7), attraction_coefficient),
                                  np.power(distance_anomaly, -13))

    return np.multiply(force_magnitude, displacement_normalized)

def create_particles(particle_number: int, dimension: int, stdev: float, charges: bool) -> list:

    cluster_size = np.power(particle_number, 1/dimension)

    particle_data = []

    for i in range(particle_number):

        radius = np.random.lognormal(mean=0.0, sigma = stdev)

        position = [np.random.uniform(-cluster_size, cluster_size) for j in range(dimension)]

        if charges:
            charge = (-1)**(i % 2)
        else:
            charge = 0

        particle_data.append([position, radius, charge])

    return particle_data

def update_particles(particle_list: list, update_size: float, force) -> list:

    l = len(particle_list)
    d = len(particle_list[0][0])

    update_list = [[0] * d] * l
    list_out = []

    for i in range(l):

        for j in range(l):

            if i != j:

                update_list[i] = np.add(update_list[i], force(particle_list[i], particle_list[j]))

    for i in range(l):

        list_out.append([np.add(particle_list[i][0],
                                np.multiply(np.divide(update_list[i],np.linalg.norm(update_list[i])),
                                            update_size))]
                        + particle_list[i][1:])


    return list_out

def plot_particles(particle_list: list, data_string: str, dimension: int):

    ft_data = fourier_transform(particle_list, dimension)

    fig, (ax_1, ax_2) = plt.subplots(2, 1, figsize=[4,8])

    # Plot axis 1

    ax_1.plot(ft_data[0], ft_data[1])
    ax_1.set_xlabel("Wavelength / d")
    ax_1.set_ylabel("Square of the Fourier Transform")
    ax_1.spines["top"].set_visible(False)
    ax_1.spines["right"].set_visible(False)

    # Plot axis 2

    x_values = [p[0][0] for p in particle_list]
    y_values = [p[0][1] for p in particle_list]
    r_values = [p[1] for p in particle_list]

    pdi = np.power((np.std(r_values)/np.mean(r_values)),2)

    ax_2.scatter(x_values, y_values, alpha=0)

    for p in particle_list:

        #Color based on charge
        color = ["tab:blue", "tab:orange", "tab:cyan"][p[2]]

        #Do some z stuff
        try:
            z = p[0][2]
            a = 0.4
        except:
            z = 0
            a = 1

        ax_2.add_patch(Circle(p[0][0:2], p[1]*0.9, zorder = z, alpha=a, color=color))

    ax_2.tick_params(bottom=False,left=False)
    ax_2.set_xticks([])
    ax_2.set_yticks([])
    ax_2.set_xlabel("Representation of the lattice, PDI = {:.2g}".format(pdi))
    fig.savefig("Nanoparticle Crystals {}.png".format(data_string), format="png", dpi=300, bbox_inches="tight")

def anneal(particle_list: list, n: int, force = lj_force):

    for i in range(n+1, 1, -1):

        particle_list = update_particles(particle_list, np.multiply(np.divide(i,n), 1), force)

    return particle_list

def clean(particle_list: list) -> list:

    positions = [p[0] for p in particle_list]
    mean_position = np.mean(positions, axis=0)
    distances = [np.linalg.norm(np.subtract(p, mean_position)) for p in positions]
    d_mean = np.mean(distances)
    d_sigma = np.std(distances)

    list_out = [p for p in particle_list
                if np.linalg.norm(np.subtract(p[0], mean_position)) < d_mean]

    return list_out

def fourier_transform(particle_list: list, dimensions: int):

    # These will be returned

    wavelengths = [np.divide(w,2) for w in np.arange(1, 5, 0.01)]
    sq_transforms = []

    # Sample 100 unit wavevectors to get an idea of light propagating through the medium

    wave_vectors = []
    for i in range(100):
        vector = []
        for j in range(dimensions):
            vector.append(np.random.normal())
        wave_vectors.append(np.divide(vector, np.linalg.norm(vector)))

    # Perform the fourier transforms

    for wavelength in wavelengths:

        wavenumber = np.divide(1, wavelength)
        ft_sum = 0

        for wave_vector in wave_vectors:

            ft_value = [0, 0]

            for position, radius, charge in particle_list:

                # Have to consider phases when summing over a single wave vector

                magnitude = np.sin(np.product([2*np.pi, radius, wavenumber]))

                dot_product = np.dot(wave_vector, position)
                phase =  np.product([2*np.pi, dot_product, wavenumber])
                ft_value = np.add(ft_value, np.multiply(magnitude, [np.cos(phase), np.sin(phase)]))

            # Sum the squared values without considering phases, when summing over wave vectors

            ft_sum += np.sum(np.power(ft_value, 2))

        sq_transforms.append(np.divide(ft_sum,np.multiply(len(wave_vectors),
                                                          len(particle_list))))

    return (wavelengths, sq_transforms)

def data_string(particle_number: int, dimensions: int, standard_deviation: float, charges: bool, iterations: int):

    if charges:
        charge_string = "Charges"
    else:
        charge_string = "No Charges"

    return "P={}, D={}, sigma={:.2g}, {}, steps={}".format(particle_number, dimensions, standard_deviation, charge_string, iterations)



if __name__ == "__main__":
    main()
