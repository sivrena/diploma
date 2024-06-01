import numpy as np
import matplotlib.pyplot as plt

# функция, которая вычисляет, какому промежутку принадлежит значение скорости, и находит значение вероятности молекулы
# с данной скоростью. edges - промежутки значений скорости. samp - вероятности. ksi - скорость молекулы.
def distribution(edges, samp, ksi):
    result = [0., 0., 0.]
    for i in range(1, len(samp)):
        if edges[i - 1] <= ksi <= edges[i]:
            result[0] = edges[i - 1]
            result[1] = edges[i]
            result[2] = samp[i - 1]
            break
    return result

def findVelocityinLayer(particle_list, stepnumber, z):
    k = 1
    border = [4.]

    coeff = 1
    x =  [(i + 1) * coeff for i in range(stepnumber // coeff)]

    dir = [0, 1, 2]

    fig = plt.figure(figsize=(18, 6))
    gx = fig.add_subplot(1, 3, 1)
    gy = fig.add_subplot(1, 3, 2)
    gz = fig.add_subplot(1, 3, 3)

    gx.set_xlabel("step")
    gx.set_ylabel("vel_x")
    gy.set_xlabel("step")
    gy.set_ylabel("vel_y")
    gz.set_xlabel("step")
    gz.set_ylabel("vel_z")

    for m in range(k):
        # average velocity
        avx = [0. for i in range(len(x))]
        avy = [0. for i in range(len(x))]
        avz = [0. for i in range(len(x))]
        nx = [0. for i in range(len(x))]
        ny = [0. for i in range(len(x))]
        nz = [0. for i in range(len(x))]

        # Построим функцию распределения молекул на внешней границе. Выбираем молекулы в пределах верхнего слоя.
        # arr_hist - значения скоростей частиц
        # edges - разбиение возможных значений скорости на интервалы
        # samp - вероятность нахождения частицы со скоростью из данного интервала
        for i in range (len(x)):
            vel_x = np.array([])
            vel_y = np.array([])
            vel_z = np.array([])
            for j in range(len(particle_list)):
                if border[m] - 1 <= particle_list[j].solpos[x[i]][2] <= border[m]:
                    vel_x = np.append(vel_x, particle_list[j].solvel[x[i]][dir[0]])
                    vel_y = np.append(vel_y, particle_list[j].solvel[x[i]][dir[1]])
                    vel_z = np.append(vel_z, particle_list[j].solvel[x[i]][dir[2]])

            arr_hist_x, edges_x = np.histogram(vel_x, bins=20)
            samp_x = arr_hist_x / (vel_x.shape[0] * np.diff(edges_x))
            arr_hist_y, edges_y = np.histogram(vel_y, bins=20)
            samp_y = arr_hist_y / (vel_y.shape[0] * np.diff(edges_y))
            arr_hist_z, edges_z = np.histogram(vel_z, bins=20)
            samp_z = arr_hist_z / (vel_z.shape[0] * np.diff(edges_z))

            average = [0., 0., 0.]

            for j in range (1, len(vel_x)):
                # С помощью функции distribution находим интервал, которому принадлежит
                # скорость, и её вероятность.
                fx = distribution(edges_x, samp_x, vel_x[j-1])
                nx[i] += fx[2] * (fx[1] - fx[0]) # n = integral(f(t,x,ksi)*dxi)
                fy = distribution(edges_y, samp_y, vel_y[j-1])
                ny[i] += fy[2] * (fy[1] - fy[0])
                fz = distribution(edges_z, samp_z, vel_z[j-1])
                nz[i] += fz[2] * (fz[1] - fz[0])

                average[0] += vel_x[j - 1] * fx[2] * (fx[1] - fx[0]) # u = 1/n * integral(ksi*f(t,x,ksi)*dxi)
                average[1] += vel_y[j - 1] * fy[2] * (fy[1] - fy[0])
                average[2] += vel_z[j - 1] * fz[2] * (fz[1] - fz[0])

            avx[i] += average[0] / nx[i]
            avy[i] += average[1] / ny[i]
            avz[i] += average[2] / nz[i]

        gx.plot(x, avx, label="Av. Vel_x on the outer border")
        gx.legend(loc="upper right")
        gy.plot(x, avy, label="Av. Vel_y on the outer border")
        gy.legend(loc="upper right")
        gz.plot(x, avz, label="Av. Vel_z on the outer border")
        gz.legend(loc="upper right")

    plt.show()