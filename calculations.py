from dataInitialization import getData, init_list_Al, init_list_Ar
from particleInteraction import solve_step
from makeDiagrams import graphResults
from KnudsenLayer import findParameters
from averageVelocity import findAverageVelocityatPoint
from distribIntegral import countIntegral
import Dump

import numpy as np
from math import pi
import os
#from decimal import Decimal

def moleculeData ():
    parameters = getData()

    Borders = [parameters['Borders'][0][1], parameters['Borders'][1][1], parameters['Borders'][2][1]]  # границы области
    tfin = 2e-3# время симуляции
    stepnumber = 200  # число шагов
    timestep = (tfin) / (stepnumber)  # временной шаг

    BoltsmanConstant = 1.38 * 10e-23
    temperature = 300  # Кельвины

    particle_number_Al = 0  # число частиц
    radius_Al = parameters['AlRadius']  # данные рассматриваемой частицы
    mass_Al = parameters['AlMass']
    epsilon_Al = parameters['AlEps']
    sigma_Al = parameters['AlSigma']
    alpha_Al = parameters['AlAlpha']

    particle_number_Ar = parameters['ArNumOfAtoms']  # число частиц
    radius_Ar = parameters['ArRadius']  # данные рассматриваемой частицы
    mass_Ar = parameters['ArMass']
    epsilon_Ar = parameters['ArEpsilon']
    sigma_Ar = parameters['ArSigma']
    alpha_Ar = 0

    particle_list_Al = init_list_Al(particle_number_Al, radius_Al, mass_Al, epsilon_Al, sigma_Al, alpha_Al, Borders)
    z = particle_list_Al[len(particle_list_Al) - 1].position[2]
    center_x = Borders[0] / 2
    center_y = Borders[1] / 2
    # rr = (particle_list_Al[len(particle_list_Al) - 1].position[0] - center_x) ** 2 + \
    #      (particle_list_Al[len(particle_list_Al) - 1].position[1] - center_y) ** 2
    center = [center_x, center_y]
    R = 1.21 # changed
    Z = z
    volume = Borders[0] * Borders[1] * (Borders[2] - z) + pi * z * z * (R - 1 / 3 * z)

    velFlow = np.array([1000., 0., 0.])
    particle_list_Ar = init_list_Ar(particle_number_Ar, radius_Ar, mass_Ar, epsilon_Ar, sigma_Ar, alpha_Ar,
                                           Borders, z, BoltsmanConstant, temperature, velFlow)

    particle_number_Al = len(particle_list_Al)
    particle_number = particle_number_Ar + particle_number_Al
    particle_list = np.concatenate([particle_list_Ar, particle_list_Al])

    # pressure_moment = np.array([0. for i in range(stepnumber)])
    # adsorption_moment = np.array([0. for i in range(stepnumber)])
    # adsorbent_number = 0
    # for i in range(particle_number_Al):
    #     if particle_list_Al[i].position[2] < z:
    #         adsorbent_number += 1
    # adsorbent_number += round(2 * pi * R / 2 * radius_Al)

    # Вычислительный эксперимент
    OutputFileName = "output.dump"
    if os.path.exists(OutputFileName):
        os.remove(OutputFileName)

    for i in range(stepnumber):

        Radius = np.array([particle_list[j].radius for j in range(len(particle_list))])
        Positions = np.array(([particle_list[j].position for j in range(len(particle_list))]))
        Velocities = np.array([particle_list[j].velocity for j in range(len(particle_list))])
        Types = np.array([particle_list[j].adsorbate for j in range(len(particle_list))])
        Dump.writeOutput(OutputFileName, particle_number, i, Borders,
                         radius=Radius, pos=Positions, velocity=Velocities, type=Types)

        adsorbed_number = 0
        solve_step(particle_list, particle_number_Ar, particle_number_Al, timestep, Borders, center, R, Z, velFlow)

        # Вычисление центра инерции и момента сил t = start or t = finish
        # if i == 0 or i == stepnumber - 1:
        #     inertia_center = inertiaCenter(particle_list[:particle_number_Ag])
        #     M = torque(inertia_center, particle_list[:particle_number_Ag])

        for particle in particle_list:
            if particle.adsorbate and particle.position[2] < Z and \
                    (particle.position[0] - center_x) ** 2 + (particle.position[1] - center_y) ** 2 < R * R:
                adsorbed_number += 1

        # pressure_moment[i] += (particle_number_Ar - adsorbed_number) / (volume * 1e-27) * BoltsmanConstant * temperature
        # adsorption_moment[i] += (adsorbed_number) / (adsorbent_number)

    findAverageVelocityatPoint(particle_list[:particle_number_Ar], stepnumber, Z)
    findParameters(particle_list[:particle_number_Ar], stepnumber, Z)
    graphResults(particle_list[:particle_number_Ar], tfin, timestep)
    #countIntegral(particle_list[:particle_number_Ar], stepnumber)

    #Запись данных, необходимых для построения изотермы адсорбции, в файл
    # result = str(sum(pressure_moment) / stepnumber) + ' ' + str(sum(adsorption_moment) / stepnumber) + ' ' + str(
    #     temperature) \
    #          + ' ' + str(particle_number_Ag) + ' ' + str(Borders[0]) + ' ' + str(Borders[1]) + ' ' + str(Borders[2])
    #
    # with open('isotherms_graph.txt', 'a') as f:
    #     f.write(result + '\n')


#рассчет средней длины пробега молекул газа с учетом движения
def meanFreePath(radius_Ar, particle_number_Ar, volume):
    free_path = 1. / (np.sqrt(2) * np.pi * np.square(2 * radius_Ar) * (particle_number_Ar / volume))
    print("Длина свободного пробега " + str(free_path))

# рассчет центра инерции
def inertiaCenter(particle_list):
    sum_m = 0.
    sum_rm = [0., 0., 0.]
    for particle in particle_list:
        sum_m += particle.mass
        sum_rm += particle.mass * particle.position
    inertia_center = sum_rm / sum_m

    return inertia_center

def torque(inertia_center, particle_list):
    M = [0., 0., 0.]
    for particle in particle_list:
        r_v = particle.position - inertia_center
        M[0] += r_v[1] * particle.force[2] - r_v[2] * particle.force[1]
        M[1] += r_v[2] * particle.force[0] - r_v[0] * particle.force[2]
        M[2] += r_v[0] * particle.force[1] - r_v[1] * particle.force[0]

    return M