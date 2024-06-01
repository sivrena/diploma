from dataInitialization import getData, init_list_Al, init_list_Ar
from particleInteraction import solve_step
from distributionDiagrams import graphResults
from averageVelocity import findVelocityinLayer
from interaction import calculateInteraction
import Dump

import numpy as np
from math import pi
import os

def moleculeData ():
    #Считывание данных из файла
    parameters = getData()

    Borders = [parameters['Borders'][0][1], parameters['Borders'][1][1], parameters['Borders'][2][1]]  # границы области
    timestep = 1e-5  # временной шаг
    stepnumber = 100 # число шагов
    tfin = timestep * stepnumber # время симуляции

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
    z1 = z + 2 * radius_Al * 2

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

        solve_step(particle_list, particle_number_Ar, particle_number_Al, timestep, Borders, center, R, z, z1)

    # Выисление средней скорости на внешней границе
    findVelocityinLayer(particle_list[:particle_number_Ar], stepnumber, z)

    # Построение распределения молекул газа по скоростям
    # graphResults(particle_list[:particle_number_Ar], tfin, timestep, z)

    # Вычисление силы взаимодействия газа с некоторыми точками поверхности
    calculateInteraction(particle_list, particle_number_Ar, stepnumber)