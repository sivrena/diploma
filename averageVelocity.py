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

# Средняя скорость молекул в данной точке пространства
def findAverageVelocityatPoint(particle_list, stepnumber, z):
    # Выбираем точку, которую будем рассматривать
    r = particle_list[0].radius
    points = [[2., 2., 3.5], [0.5, 2., 2.], [2., 2., z + 2*r], [2., 2., 2.], [0.5, 2., 3.5]]
    # Объем куба вокруг рассматриваемой точки
    cubeside = [2., 2., 2.]

    x = [i for i in range(stepnumber)]

    fig = plt.figure(figsize=(18, 5))
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

                # Добавляем в массив частицы, попавшие в область рассматриваемого куба.
                if f:
                    cubevel_x = np.append(cubevel_x, particle_list[j].solvel[stepnumber][0])
                    cubevel_y = np.append(cubevel_y, particle_list[j].solvel[stepnumber][1])
                    cubevel_z = np.append(cubevel_z, particle_list[j].solvel[stepnumber][2])

            # Построим функцию распределения молекул внутри куба. arr_hist - значения скоростей частиц
            # edges - разбиение возможных значений скорости на интервалы
            # samp - вероятность нахождения частицы со скоростью из данного интервала
            arr_hist_x, edges_x = np.histogram(cubevel_x, bins=20)
            samp_x = arr_hist_x / (cubevel_x.shape[0] * np.diff(edges_x))
            arr_hist_y, edges_y = np.histogram(cubevel_y, bins=20)
            samp_y = arr_hist_y / (cubevel_y.shape[0] * np.diff(edges_y))
            arr_hist_z, edges_z = np.histogram(cubevel_z, bins=20)
            samp_z = arr_hist_z / (cubevel_z.shape[0] * np.diff(edges_z))

            average = [0., 0., 0.]

            for j in range (1, len(cubevel_x)):
                # Вычисляем число частиц в кубике. С помощью функции distribution находим интервал, которому принадлежит
                # скорость, и её вероятность.
                fx = distribution(edges_x, samp_x, cubevel_x[j - 1])
                nx[i] += fx[2] * (fx[1] - fx[0]) # n = integral(f(t,x,ksi)*dxi)
                fy = distribution(edges_y, samp_y, cubevel_y[j - 1])
                ny[i] += fy[2] * (fy[1] - fy[0])
                fz = distribution(edges_z, samp_z, cubevel_z[j - 1])
                nz[i] += fz[2] * (fz[1] - fz[0])

                # Средняя скорость в окрестности рассматриваемой точки
                # Значение скорости умножаем на её вероятность и интервал, которому она принадлежит
                # Далее делим на число частиц
                average[0] += cubevel_x[j - 1] * fx[2] * (fx[1] - fx[0]) # u = 1/n * integral(ksi*f(t,x,ksi)*dxi)
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

def findVelocityinLayer(particle_list, stepnumber):
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