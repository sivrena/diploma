from particleInteraction import Particle

import numpy as np
from math import pi, sin, cos
from pandas import read_csv
import json

# Считывание данных, необходимых для моделирования, из файла
def getData ():
    parameters = {
        'ArNumOfAtoms': 0,
        'ArRadius': 0,
        'ArMass': 0,
        'ArEpsilon': 0,
        'ArSigma': 0,

        'AlNumOfAtoms': 0,
        'AlRadius': 0,
        'AlMass': 0,
        'AlEpsilon': 0,
        'AlSigma': 0,
        'AlEps': 0,
        'AlAlpha': 0,

        'TimeStep': 1e-16,
        'Steps': 100,
        'OutputFrequency': 2,
        'Borders': [[], [], []],
        'OutputFileName': 'output.dump'
    }

    dataObj = read_csv('Data.csv', delimiter=';')
    csvList = [tuple(row) for row in dataObj.values]
    for x in csvList:
        if x[0] == 'Borders':
            parameters[x[0]] = json.loads(x[1])
        elif x[0] == 'OutputFileName':
            parameters[x[0]] = x[1]
        else:
            parameters[x[0]] = float(x[1])
    parameters['ArNumOfAtoms'] = int(parameters['ArNumOfAtoms'])
    parameters['Steps'] = int(parameters['Steps'])
    parameters['OutputFrequency'] = int(parameters['OutputFrequency'])

    return parameters

# Моделирование твердой поверхности
def init_list_Al(N, radius, mass, epsilon, sigma, alpha, borders):
    particle_list = []
    particle_position_x = np.array([])
    particle_position_y = np.array([])
    particle_position_z = np.array([])

    particle_number_bottom = np.array([1])
    for i in range (1):
        particle_number_bottom = np.append(particle_number_bottom, int(round(2 * pi * radius * (i + 2) / (2 * radius))))
    angle_bottom = [2 * pi / particle_number_bottom[i] for i in range(len(particle_number_bottom))]
    radius_bottom = [radius * (i * 1.1 + 1) for i in range(len(particle_number_bottom))]

    particle_number_layers = np.array([])
    for i in range (4):
        particle_number_layers = np.append(particle_number_layers, int(round(2 * pi * radius *\
                                                                (i + 1 + len(particle_number_bottom)) / (2 * radius))))
    angle_layers = [2 * pi / particle_number_layers[i] for i in range(len(particle_number_layers))]
    radius_layers = [radius * (i + len(particle_number_bottom) + 1) for i in range(len(particle_number_layers))]

    center_x = borders[0] / 2
    center_y = borders[1] / 2
    particle_position_x = np.append(particle_position_x, center_x)
    particle_position_y = np.append(particle_position_y, center_y)
    particle_position_z = np.append(particle_position_z, 0.)

    # surface = np.array([31, 35, 40, 45, 50, 55])
    # angle_surface = [2 * pi / surface[i] for i in range(len(surface))]
    # radius_surface = [radius * (i + len(particle_number_bottom) + len(particle_number_layers) + 1) for i in
    #                   range(len(surface))]

    for i in range(1, len(particle_number_bottom)):
        r = radius_bottom[i]
        angle = angle_bottom[i]
        for j in range(particle_number_bottom[i]):
            particle_position_x = np.append(particle_position_x, center_x + r * cos(angle))
            particle_position_y = np.append(particle_position_y, center_y + r * sin(angle))
            particle_position_z = np.append(particle_position_z, 0.)
            angle += angle_bottom[i]

    for i in range(len(particle_number_layers)):
        r = radius_layers[i]
        angle = angle_layers[i]
        for j in range(int(particle_number_layers[i])):
            particle_position_x = np.append(particle_position_x, center_x + r * cos(angle))
            particle_position_y = np.append(particle_position_y, center_y + r * sin(angle))
            particle_position_z = np.append(particle_position_z, (i + 1) * 1.5 * radius)
            angle += angle_layers[i]

    z = particle_position_z[(len(particle_position_z) - 1)] + radius * 1.25
    particle_in_row_x = 15
    particle_in_row_y = 15
    kx = borders[0] / particle_in_row_x
    ky = borders[1] / particle_in_row_y

    z1 = z
    for i in range (2):
        z1 += radius * 2
        for j in range (particle_in_row_x + 1):
            x = j * kx
            y = 0
            particle_position_x = np.append(particle_position_x, x)
            particle_position_y = np.append(particle_position_y, y)
            particle_position_z = np.append(particle_position_z, z1)

            y = borders[1]
            particle_position_x = np.append(particle_position_x, x)
            particle_position_y = np.append(particle_position_y, y)
            particle_position_z = np.append(particle_position_z, z1)
        for j in range (particle_in_row_y + 1):
            y = j * ky
            x = 0
            particle_position_x = np.append(particle_position_x, x)
            particle_position_y = np.append(particle_position_y, y)
            particle_position_z = np.append(particle_position_z, z1)

            x = borders[0]
            particle_position_x = np.append(particle_position_x, x)
            particle_position_y = np.append(particle_position_y, y)
            particle_position_z = np.append(particle_position_z, z1)


    for i in range (particle_in_row_x + 1):
        for j in range (particle_in_row_y + 1):
            x = i * kx
            y = j * ky
            if (x - center_x) ** 2 + (y - center_y) ** 2 > r ** 2:
                particle_position_x = np.append(particle_position_x, x)
                particle_position_y = np.append(particle_position_y, y)
                particle_position_z = np.append(particle_position_z, z)

    # k = 1.0
    # for i in range(len(surface)):
    #     r = radius_surface[i]
    #     angle = angle_surface[i]
    #     for j in range(int(surface[i])):
    #         x = center_x + r * k * cos(angle)
    #         y = center_y + r * k * sin(angle)
    #         if (x <= borders[0] and y <= borders[1] and z<= borders[2] and x >= 0 and y >= 0 and z >= 0):
    #             particle_position_x = np.append(particle_position_x, x)
    #             particle_position_y = np.append(particle_position_y, y)
    #             particle_position_z = np.append(particle_position_z, z)
    #         angle += angle_surface[i]
    #     k += 0.05

    N += len(particle_position_x)
    for i in range(N):
        v = np.array([0., 0., 0.])
        f = np.array([0. for i in range (len(v))])
        a = np.array([0. for i in range(len(v))])
        pos = np.array([particle_position_x[i], particle_position_y[i], particle_position_z[i]])
        newparticle = Particle(mass, radius, epsilon, sigma, pos, v, f, a, alpha, adsorbate=0)

        particle_list.append(newparticle)
    return particle_list

def init_list_Ar (N, radius, mass, epsilon, sigma, alpha, borders, z, k, T, flow):
    # Случайным образом генерируем массив объектов Particle, число частиц равно N
    # В данной программе рассмотрен трехмерный случай
    particle_list = []

    dim = 3
    std_dev = np.sqrt(k * T / mass)

    for i in range(N):
        f = np.array([0. for i in range (dim)])
        a = np.array([0. for i in range(dim)])

        # Генерирование начальных скоростей с помощью распределения Максвелла-Больцмана
        # velx = np.random.normal(loc=0, scale=std_dev)
        # vely = np.random.normal(loc=0, scale=std_dev)
        # velz = np.random.normal(loc=0, scale=std_dev)
        velx = 0.
        vely = 0.
        velz = 0.

        collision = True
        while (collision == True):
            collision = False
            # Генерирование позиций частиц газа через равномерное нормальное распределение
            posx = radius + np.random.uniform(low=0., high=4.)
            if posx < 0. + radius:
                posx += radius
            elif posx > borders[0] - radius:
                posx -= radius

            posy = radius + np.random.uniform(low=0., high=4.)
            if posy < 0. + radius:
                posy += radius
            elif posy > borders[1] - radius:
                posy -= radius
            posz = radius + np.random.uniform(low=(z + 2 * radius), high=(4. - 2 * radius))
            pos = np.array([posx, posy, posz])
            v = np.array([velx, vely, velz])

            newparticle = Particle(mass, radius, epsilon, sigma, pos, v, f, a, alpha, adsorbate=1)
            newparticle.flow_vel += flow

            k = 1
            h = borders[1]
            y = 1.45
            newparticle.velocity = flow * k / h * y # задание направленной скорости
            for j in range(len(particle_list)):
                collision = newparticle.check_coll(particle_list[j])
                if collision == True:
                    break

        particle_list.append(newparticle)
    return particle_list