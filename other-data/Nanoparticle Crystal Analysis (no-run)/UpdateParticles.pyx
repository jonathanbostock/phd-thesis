# Yeah cython go cython

def anneal(particle_list: list, n: int) -> list:

    output_list: list = particle_list

    for i in range(n+1, 1, -1):

        update_size: float = float(i)/float(n)
        output_list = update_particles(particle_list, update_size)

    return output_list

def lj_force(p_1: list, p_2: list,
             r_1: float, r_2: float,
             c_1: int, c_2: int) -> list:

    d: int = len(p_1)

    displacement: list = [p_2[k] - p_1[k] for k in range(d)]
    distance: float = norm(displacement)
    distance_anomaly: float = distance / (r_1 + r_2)
    displacement_normalized: list = [displacement[k] / distance for k in range(d)]

    # This bit is kinda fucked, but it lets you get a coefficient quickly with indexing
    # If both charges are zero, attraction occurs
    # If they are 1 and -1, attraction occurs
    # Else no attraction
    attraction_coefficient: float = ((1.0, 0.1, 1.0),
                                     (0.1, 0.1, 1.0),
                                     (0.1, 1.0, 0.1))[c_1][c_2]

    force_magnitude: float = attraction_coefficient/(distance_anomaly ** 7) - 1/(distance_anomaly ** 13)

    return_vector: list = [(force_magnitude * displacement_normalized[k]) for k in range(d)]

    return return_vector

def norm(vector: list) -> float:

    sq_output: float = 0
    for item in vector:
        sq_output += item ** 2

    return sq_output ** (1/2)

def update_particles(particle_list: list, update_size: float) -> list:

    l: int = len(particle_list)
    d: int = len(particle_list[0][0])

    update_list: list = [[0] * d] * l
    list_out: list = []

    for  i in range(l):

        for j in range(l):

            if i != j:

                force_output = lj_force(particle_list[i][0], particle_list[j][0],
                                        particle_list[i][1], particle_list[j][1],
                                        particle_list[i][2], particle_list[j][2])

                update_list[i] = [update_list[i][k] + force_output[k] for k in range(d)]

    for i in range(l):

        list_out.append([particle_list[i][0][k] + ((update_list[i][k] / norm(update_list[i])) * update_size) for k in range(d)] + particle_list[i][1:])

    return list_out
