import numpy as np
import matplotlib.pyplot as plt

def findParameters(particle_list, stepnumber, z):

    r_g = 0.145 #эффективный радиус взаимодействия газа
    k = 1
    #thick = [r_g * (i + 1) for i in range (k)]
    border = [4.]

    coeff = 1
    x =  [(i + 1) * coeff for i in range(stepnumber // coeff)]

    dir = [0, 1, 2]

    # Graph average velocity in Knudsen layer
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

        # average integral velocity
        aivx = [0. for i in range(len(x))]
        aivy = [0. for i in range(len(x))]
        aivz = [0. for i in range(len(x))]

        for i in range (len(x)):
            vel_x = np.array([])
            vel_y = np.array([])
            vel_z = np.array([])
            for j in range(len(particle_list)):
                if particle_list[j].solpos[x[i]][2] <= border[m]:
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

            for j in range (1, len(samp_x)):
                nx[i] += samp_x[j - 1] * (edges_x[j] - edges_x[j - 1])
                ny[i] += samp_y[j - 1] * (edges_y[j] - edges_y[j - 1])
                nz[i] += samp_z[j - 1] * (edges_z[j] - edges_z[j - 1])
                average[0] += (edges_x[j] + edges_x[j - 1]) / 2 * samp_x[j - 1] * (edges_x[j] - edges_x[j - 1])
                average[1] += (edges_y[j] + edges_y[j - 1]) / 2 * samp_y[j - 1] * (edges_y[j] - edges_y[j - 1])
                average[2] += (edges_z[j] + edges_z[j - 1]) / 2 * samp_z[j - 1] * (edges_z[j] - edges_z[j - 1])

            avx[i] += average[0] / nx[i]
            avy[i] += average[1] / ny[i]
            avz[i] += average[2] / nz[i]

        gx.plot(x, avx, label="Av. Vel_x in layer " + str(round(border[m], 2)) + "nm")
        gx.legend(loc="upper right")
        gy.plot(x, avy, label="Av. Vel_y in layer " + str(round(border[m], 2)) + "nm")
        gy.legend(loc="upper right")
        gz.plot(x, avz, label="Av. Vel_z in layer " + str(round(border[m], 2)) + "nm")
        gz.legend(loc="upper right")

    plt.show()