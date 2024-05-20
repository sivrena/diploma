from dataInitialization import getData, init_list_Al, init_list_Ar
from particleInteraction import solve_step
from distributionDiagrams import graphResults
from averageVelocity import findVelocityinLayer
import Dump

import numpy as np
from math import pi
import os

def moleculeData ():
    #Считывание данных из файла
    parameters = getData()

    Borders = [parameters['Borders'][0][1], parameters['Borders'][1][1], parameters['Borders'][2][1]]  # границы области
    tfin = 3.5e-3# время симуляции
    stepnumber = 350  # число шагов
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
    center = [center_x, center_y]
    R = 1.21
    Z = z
    volume = Borders[0] * Borders[1] * (Borders[2] - z) + pi * z * z * (R - 1 / 3 * z)

    velFlow = np.array([1000., 0., 0.])
    particle_list_Ar = init_list_Ar(particle_number_Ar, radius_Ar, mass_Ar, epsilon_Ar, sigma_Ar, alpha_Ar,
                                           Borders, z, BoltsmanConstant, temperature, velFlow)

    particle_number_Al = len(particle_list_Al)
    particle_number = particle_number_Ar + particle_number_Al
    particle_list = np.concatenate([particle_list_Ar, particle_list_Al])

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

        for particle in particle_list:
            if particle.adsorbate and particle.position[2] < Z and \
                    (particle.position[0] - center_x) ** 2 + (particle.position[1] - center_y) ** 2 < R * R:
                adsorbed_number += 1

    # Выисление средней скорости в точке
    # findAverageVelocityatPoint(particle_list[:particle_number_Ar], stepnumber, Z)
    # Выисление средней скорости на внешней границе
    findVelocityinLayer(particle_list[:particle_number_Ar], stepnumber)
    # Построение распределения молекул газа по скоростям
    graphResults(particle_list[:particle_number_Ar], tfin, timestep)