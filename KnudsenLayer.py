import numpy as np
import matplotlib.pyplot as plt

def findParameters(particle_list, stepnumber, z):

    r_g = 0.145 #эффективный радиус взаимодействия газа
    k = 1
    thick = [r_g * (i + 1) for i in range (k)]
    border = [z + thick[i] for i in range (k)]

    x =  [(i + 1)  * 1 for i in range(stepnumber // 1)]
    avx = [0. for i in range(len(x))]
    avy = [0. for i in range(len(x))]
    avz = [0. for i in range(len(x))]
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
        for i in range (len(x)):
            vel_x = np.array([])
            vel_y = np.array([])
            vel_z = np.array([])
            for j in range(len(particle_list)):
                if particle_list[j].solpos[x[i]][2] < border[m]:
                    vel_x = np.append(vel_x, particle_list[j].solvel[x[i]][dir[0]])
                    vel_y = np.append(vel_y, particle_list[j].solvel[x[i]][dir[1]])
                    vel_z = np.append(vel_z, particle_list[j].solvel[x[i]][dir[2]])
            hist_vx, edges_vx = np.histogram(vel_x, bins=20)
            hist_vy, edges_vy = np.histogram(vel_y, bins=20)
            hist_vz, edges_vz = np.histogram(vel_z, bins=20)

            general = [sum(hist_vx), sum(hist_vy), sum(hist_vz)]
            average = [0., 0., 0.]

            for j in range (1, len(edges_vx)):
                average[0] += (edges_vx[j] + edges_vx[j - 1]) / 2 * (hist_vx[j - 1] / general[0])
                average[1] += (edges_vy[j] + edges_vy[j - 1]) / 2 * (hist_vy[j - 1] / general[1])
                average[2] += (edges_vz[j] + edges_vz[j - 1]) / 2 * (hist_vz[j - 1] / general[2])

            avx[i] += average[0]
            avy[i] += average[1]
            avz[i] += average[2]
            # print('step ' + str(x[i]))
            # print(str(average[0]) + '    ' + str(average[1]) + '    ' + str(average[2]) + '\n\n')

        gx.plot(x, avx, label="Average Vel_x")
        gx.legend(loc="upper right")
        gy.plot(x, avy, label="Average Vel_y")
        gy.legend(loc="upper right")
        gz.plot(x, avz, label="Average Vel_z")
        gz.legend(loc="upper right")
        plt.show()