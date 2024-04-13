import numpy as np
import matplotlib.pyplot as plt

def findAvIntVel(hist_vi, edges_vi):
    general = sum(hist_vi)

    sum_neg = 0.
    N_neg = 0
    sum_pos = 0.
    N_pos = 0

    for i in (range(len(hist_vi))):
        if edges_vi[i] <= 0 and edges_vi[i+1] <= 0:
            sum_neg += hist_vi[i] / general
            N_neg += 1
        elif edges_vi[i] >= 0 and edges_vi[i+1] >= 0:
            sum_pos += hist_vi[i] / general
            N_pos += 1
        else:
            sum_neg += hist_vi[i] / general
            N_neg += 1
            sum_pos += hist_vi[i] / general
            N_pos += 1

    if N_pos > 0:
        v_pos = sum_pos / N_pos
    else:
        v_pos = 0.

    if N_neg > 0:
        v_neg = sum_neg / N_neg
    else:
        v_neg = 0.

    return v_pos, v_neg

def findParameters(particle_list, stepnumber, z):

    r_g = 0.145 #эффективный радиус взаимодействия газа
    k = 3
    thick = [r_g * (i + 1) for i in range (k)]
    border = [z + thick[i] for i in range (k)]

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

    # Graph average INTEGRAL velocity in Knudsen layer (molecule vel > 0)
    fig1 = plt.figure(figsize=(18, 6))
    ivx_pos = fig1.add_subplot(1, 3, 1)
    ivy_pos = fig1.add_subplot(1, 3, 2)
    ivz_pos = fig1.add_subplot(1, 3, 3)

    ivx_pos.set_xlabel("step")
    ivx_pos.set_ylabel("average integral velocity_x")
    ivy_pos.set_xlabel("step")
    ivy_pos.set_ylabel("average integral velocity_y")
    ivz_pos.set_xlabel("step")
    ivz_pos.set_ylabel("average integral velocity_z")

    # Graph average INTEGRAL velocity in Knudsen layer (molecule vel < 0)
    fig2 = plt.figure(figsize=(18, 6))
    ivx_neg = fig2.add_subplot(1, 3, 1)
    ivy_neg = fig2.add_subplot(1, 3, 2)
    ivz_neg = fig2.add_subplot(1, 3, 3)

    ivx_neg.set_xlabel("step")
    ivx_neg.set_ylabel("average integral velocity_x")
    ivy_neg.set_xlabel("step")
    ivy_neg.set_ylabel("average integral velocity_y")
    ivz_neg.set_xlabel("step")
    ivz_neg.set_ylabel("average integral velocity_z")

    for m in range(k):
        # average velocity
        avx = [0. for i in range(len(x))]
        avy = [0. for i in range(len(x))]
        avz = [0. for i in range(len(x))]

        # average integral velocity
        paivx = [0. for i in range(len(x))]
        paivy = [0. for i in range(len(x))]
        paivz = [0. for i in range(len(x))]

        naivx = [0. for i in range(len(x))]
        naivy = [0. for i in range(len(x))]
        naivz = [0. for i in range(len(x))]

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
            intVel_x = findAvIntVel(hist_vx, edges_vx)
            hist_vy, edges_vy = np.histogram(vel_y, bins=20)
            intVel_y = findAvIntVel(hist_vy, edges_vy)
            hist_vz, edges_vz = np.histogram(vel_z, bins=20)
            intVel_z = findAvIntVel(hist_vz, edges_vz)

            general = [sum(hist_vx), sum(hist_vy), sum(hist_vz)]
            average = [0., 0., 0.]

            for j in range (1, len(edges_vx)):
                average[0] += (edges_vx[j] + edges_vx[j - 1]) / 2 * (hist_vx[j - 1] / general[0])
                average[1] += (edges_vy[j] + edges_vy[j - 1]) / 2 * (hist_vy[j - 1] / general[1])
                average[2] += (edges_vz[j] + edges_vz[j - 1]) / 2 * (hist_vz[j - 1] / general[2])

            avx[i] += average[0]
            avy[i] += average[1]
            avz[i] += average[2]

            paivx[i] += intVel_x[0]
            paivy[i] += intVel_y[0]
            paivz[i] += intVel_z[0]

            naivx[i] += intVel_x[1]
            naivy[i] += intVel_y[1]
            naivz[i] += intVel_z[1]
            # print('step ' + str(x[i]))
            # print(str(average[0]) + '    ' + str(average[1]) + '    ' + str(average[2]) + '\n\n')

        gx.plot(x, avx, label="Av. Vel_x in Kn. layer " + str(round(thick[m],2)) + "nm")
        gx.legend(loc="upper right")
        gy.plot(x, avy, label="Av. Vel_y in Kn. layer " + str(round(thick[m],2)) + "nm")
        gy.legend(loc="upper right")
        gz.plot(x, avz, label="Av. Vel_z in Kn. layer " + str(round(thick[m],2)) + "nm")
        gz.legend(loc="upper right")

        ivx_pos.plot(x, paivx, label="Av. int. vel_x\nParticle vel. > 0\nKn. layer " + str(round(thick[m],2)) + "nm")
        ivx_pos.legend(loc="upper right")
        ivy_pos.plot(x, paivy, label="Av. int. vel_y\nParticle vel. > 0\nKn. layer " + str(round(thick[m], 2)) + "nm")
        ivy_pos.legend(loc="upper right")
        ivz_pos.plot(x, paivz, label="Av. int. vel_z\nParticle vel. > 0\nKn. layer " + str(round(thick[m], 2)) + "nm")
        ivz_pos.legend(loc="upper right")

        ivx_neg.plot(x, naivx, label="Av. int. vel_x\nParticle vel. < 0\nKn. layer " + str(round(thick[m], 2)) + "nm")
        ivx_neg.legend(loc="upper right")
        ivy_neg.plot(x, naivy, label="Av. int. vel_y\nParticle vel. < 0\nKn. layer " + str(round(thick[m], 2)) + "nm")
        ivy_neg.legend(loc="upper right")
        ivz_neg.plot(x, naivz, label="Av. int. vel_z\nParticle vel. < 0\nKn. layer " + str(round(thick[m], 2)) + "nm")
        ivz_neg.legend(loc="upper right")

    plt.show()