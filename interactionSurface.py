import numpy as np
import matplotlib.pyplot as plt

#Вычисление энергии взаимодейтсвия газа с поверхностью с помощью потенциала Леннарда-Джонса
def LennardJones (particle_list, num, stepnumber):
    force_LJ = np.array([[particle_list[i].solforce[stepnumber][0], particle_list[i].solforce[stepnumber][1],
                          particle_list[i].solforce[stepnumber][2]] for i in range (len(particle_list))])

    for i in range(num):
        for j in range(i + 1, len(particle_list)):

            if particle_list[i].adsorbate > 0 and particle_list[j].adsorbate > 0:
                sigma = particle_list[i].sigma
                epsilon = particle_list[i].epsilon

            else:
                sigma = np.sqrt(particle_list[i].sigma * particle_list[j].sigma)
                epsilon = np.sqrt(particle_list[i].epsilon * particle_list[j].epsilon)

            r = np.sqrt(np.sum(np.square(particle_list[i].solpos[stepnumber] - particle_list[j].solpos[stepnumber])))

            if r < particle_list[i].radius + particle_list[j].radius:
                r = particle_list[i].radius + particle_list[j].radius

            if r < 2.5 * sigma:
                force = (48.) * (epsilon) * ((sigma ** 12) / ((r) ** 13) - (0.5) * (sigma ** 6) / ((r)** 7))
            else: force = 0.

            vel = -(particle_list[i].solpos[stepnumber] - particle_list[j].solpos[stepnumber]) * force / r

            force_LJ[i][0] += vel[0]
            force_LJ[i][1] += vel[1]
            force_LJ[i][2] += vel[2]
            force_LJ[j][0] -= vel[0]
            force_LJ[j][1] -= vel[1]
            force_LJ[j][2] -= vel[2]

    return force_LJ

def calculateInteraction(particle_list, Ar_num, steps):
    # функция на вход получает массив с данными о всех частицах в системе
    # выбрано несколько молекул алюминия: внутри поры, на поверхности, на границе поры
    Al_numbers = [71, 122, 228, 303, 317, 373]

    fig = plt.figure(figsize=(12, 12))
    gx = fig.add_subplot(2, 2, 1)
    gy = fig.add_subplot(2, 2, 2)
    gz = fig.add_subplot(2, 2, 3)
    g = fig.add_subplot(2, 2, 4)

    gx.set_xlabel("step")
    gx.set_ylabel("interaction force x, U(r)")
    gy.set_xlabel("step")
    gy.set_ylabel("interaction force y, U(r)")
    gz.set_xlabel("step")
    gz.set_ylabel("interaction force z, U(r)")
    g.set_xlabel("step")
    g.set_ylabel("interaction force module, U(r)")

    # Вычисление силы взаимодействия, построение графиков
    for j in range (len(Al_numbers)):
        particles = np.concatenate([particle_list[:Ar_num], [particle_list[Al_numbers[j]]]])

        x = [i for i in range(steps)]
        fx = [0. for i in range(steps)]
        fy = [0. for i in range(steps)]
        fz = [0. for i in range(steps)]
        f = [0. for i in range(steps)]

        for i in range (steps):
            forces = LennardJones(particles, Ar_num, i)
            fx[i] += forces[len(particles) - 1][0]
            fy[i] += forces[len(particles) - 1][1]
            fz[i] += forces[len(particles) - 1][2]
            f[i] += sum(np.square(forces[len(particles)-1]))

        lbl = "(" + str(round(particle_list[Al_numbers[j]].position[0], 2)) + ", " \
              + str(round(particle_list[Al_numbers[j]].position[1], 2)) + ", " \
              + str(round(particle_list[Al_numbers[j]].position[2], 2)) + ")"
        gx.plot(x, fx, label=lbl)
        gx.legend()
        gy.plot(x, fy, label=lbl)
        gy.legend()
        gz.plot(x, fz, label=lbl)
        gz.legend()
        g.plot(x, f, label=lbl)
        g.legend()

    plt.show()