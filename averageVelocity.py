import numpy as np
import matplotlib.pyplot as plt

def distribution(edges, samp, ksi):
    result = [0., 0., 0.]
    for i in range(1, len(samp)):
        if ksi >= edges[i - 1] and ksi <= edges[i]:
            result[0] = edges[i - 1]
            result[1] = edges[i]
            result[2] = samp[i - 1]
            break
    return result

def findAverageVelocityatPoint(particle_list, stepnumber, z):
    # Выбираем точку, которую будем рассматривать
    r = particle_list[0].radius
    points = [[2., 2., 3.5], [0.5, 2., 2.], [2., 2., z + 2*r], [2., 2., 2.]]
    cubeside = [2., 2., 2.]

    x = [i for i in range(stepnumber)]

    fig = plt.figure(figsize=(20, 5))
    gx = fig.add_subplot(1, 3, 1)
    gy = fig.add_subplot(1, 3, 2)
    gz = fig.add_subplot(1, 3, 3)

    gx.set_xlabel("step")
    gx.set_ylabel("vel_x")
    gy.set_xlabel("step")
    gy.set_ylabel("vel_y")
    gz.set_xlabel("step")
    gz.set_ylabel("vel_z")

    for k in range(len(points)):
        avx = [0. for i in range(stepnumber)]
        avy = [0. for i in range(stepnumber)]
        avz = [0. for i in range(stepnumber)]
        nx = [0. for i in range(stepnumber)]
        ny = [0. for i in range(stepnumber)]
        nz = [0. for i in range(stepnumber)]

        # Для каждого временного шага 1) определяем функцию распределения по данным моделирования
        # 2) Определяем частицы, попавшие в интересующий нас объем
        # 3) Считаем интересующие величины (среднюю скорость)
        for i in range(stepnumber):
            vel_x = np.array([])
            vel_y = np.array([])
            vel_z = np.array([])
            cubevel_x = np.array([])
            cubevel_y = np.array([])
            cubevel_z = np.array([])
            for j in range(len(particle_list)):
                f = True
                if particle_list[j].solpos[stepnumber][0] > (points[k][0] + 1. / 2 * cubeside[0])\
                        or particle_list[j].solpos[stepnumber][0] < (points[k][0] - 1. / 2 * cubeside[0]):
                    f = False
                if particle_list[j].solpos[stepnumber][1] > (points[k][1] + 1. / 2 * cubeside[1]) or \
                        particle_list[j].solpos[stepnumber][1] < (points[k][1] - 1. / 2 * cubeside[1]):
                    f = False
                if particle_list[j].solpos[stepnumber][2] > (points[k][2] + 1. / 2 * cubeside[2]) or \
                        particle_list[j].solpos[stepnumber][2] < (points[k][2] - 1. / 2 * cubeside[2]):
                    f = False

                # For Distribution function
                vel_x = np.append(vel_x, particle_list[j].solvel[stepnumber][0])
                vel_y = np.append(vel_y, particle_list[j].solvel[stepnumber][1])
                vel_z = np.append(vel_z, particle_list[j].solvel[stepnumber][2])

                # Particle inside cube around the point
                if f:
                    cubevel_x = np.append(cubevel_x, particle_list[j].solvel[stepnumber][0])
                    cubevel_y = np.append(cubevel_y, particle_list[j].solvel[stepnumber][1])
                    cubevel_z = np.append(cubevel_z, particle_list[j].solvel[stepnumber][2])

            arr_hist_x, edges_x = np.histogram(vel_x, bins=20)
            samp_x = arr_hist_x / (vel_x.shape[0] * np.diff(edges_x))
            arr_hist_y, edges_y = np.histogram(vel_y, bins=20)
            samp_y = arr_hist_y / (vel_y.shape[0] * np.diff(edges_y))
            arr_hist_z, edges_z = np.histogram(vel_z, bins=20)
            samp_z = arr_hist_z / (vel_z.shape[0] * np.diff(edges_z))

            average = [0., 0., 0.]

            for j in range (1, len(cubevel_x)):
                # Число частиц в кубике
                fx = distribution(edges_x, samp_x, cubevel_x[j])
                nx[i] += fx[2] * (fx[1] - fx[0])
                fy = distribution(edges_y, samp_y, cubevel_y[j])
                ny[i] += fy[2] * (fy[1] - fy[0])
                fz = distribution(edges_z, samp_z, cubevel_z[j])
                nz[i] += fz[2] * (fz[1] - fz[0])

                # Средняя скорость в окрестности расссматриваемой точки
                average[0] += cubevel_x[j - 1] * fx[2] * (fx[1] - fx[0])
                average[1] += cubevel_y[j - 1] * fy[2] * (fy[1] - fy[0])
                average[2] += cubevel_z[j - 1] * fz[2] * (fz[1] - fz[0])

            if nx[i] == 0.:
                avx[i] = 0.
            else:
                avx[i] += average[0] / nx[i]
            if ny[i] == 0.:
                avy[i] = 0.
            else:
                avy[i] += average[1] / ny[i]
            if nz[i] == 0.:
                avz[i] = 0.
            else:
                avz[i] += average[2] / nz[i]

        gx.plot(x, avx, label="Av. Vel_x, point (" + str(points[k][0]) + ' ' + str(points[k][1]) + ' ' + str(points[k][2]) +")")
        gx.legend(loc="upper right")
        gy.plot(x, avy, label="Av. Vel_y, point (" + str(points[k][0]) + ' ' + str(points[k][1]) + ' ' + str(points[k][2]) +")")
        gy.legend(loc="upper right")
        gz.plot(x, avz, label="Av. Vel_z, point (" + str(points[k][0]) + ' ' + str(points[k][1]) + ' ' + str(points[k][2]) +")")
        gz.legend(loc="upper right")

    plt.show()