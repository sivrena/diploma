import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def total_Energy(particle_list, index):
    return sum([particle_list[i].mass / 2. * particle_list[i].solvel_mag[index] ** 2 for i in range(len(particle_list))])

def graphResults(particle_list, tfin, timestep):
    fig = plt.figure(figsize=(12, 12))

    hist = fig.add_subplot(2, 2, 1)
    hist_x = fig.add_subplot(2, 2, 2)
    hist_y = fig.add_subplot(2, 2, 3)
    hist_z = fig.add_subplot(2, 2, 4)

    plt.subplots_adjust(bottom=0.2, left=0.15)

    vel_x = np.array([])
    vel_y = np.array([])
    vel_z = np.array([])
    vel_mod = np.array([])
    for j in range(len(particle_list)):
        if 3 <= particle_list[j].solpos[0][2] <= 4:
            vel_x = np.append(vel_x, particle_list[j].solvel[0][0])
            vel_y = np.append(vel_y, particle_list[j].solvel[0][1])
            vel_z = np.append(vel_z, particle_list[j].solvel[0][2])
            vel_mod = np.append(vel_mod, particle_list[j].solvel_mag[0])

    # Graph Particles velocity[i] histogram
    dir = 0
    let = 'x'
    #vel_x = [particle_list[i].solvel[0][dir] for i in range(len(particle_list))]
    hist_x.hist(vel_x, bins=20, density=True, label="Simulation Data")
    hist_x.set_xlabel("Vel_" + let)
    hist_x.set_ylabel("Frecuency Density")

    dir = 1
    let = 'y'
    #vel_y = [particle_list[i].solvel[0][dir] for i in range(len(particle_list))]
    hist_y.hist(vel_y, bins=20, density=True, label="Simulation Data")
    hist_y.set_xlabel("Vel_" + let)
    hist_y.set_ylabel("Frecuency Density")

    dir = 2
    let = 'z'
    #vel_z = [particle_list[i].solvel[0][dir] for i in range(len(particle_list))]
    hist_z.hist(vel_z, bins=20, density=True, label="Simulation Data")
    hist_z.set_xlabel("Vel_" + let)
    hist_z.set_ylabel("Frecuency Density")

    # Graph Particles speed histogram
    #vel_mod = [particle_list[i].solvel_mag[0] for i in range(len(particle_list))]
    hist.hist(vel_mod, bins=20, density=True, label="Simulation Data")
    hist.set_xlabel("Speed")
    hist.set_ylabel("Frecuency Density")
    # hist.set_xlim([0, 1500])
    # hist.set_ylim([0, 0.01])

    # Graph Maxwell–Boltzmann distribution
    E = total_Energy(particle_list, 0)
    Average_E = E / len(particle_list)
    k = 1.38064852e-23
    T = 2 * Average_E / (2 * k)
    m = particle_list[0].mass
    v = np.linspace(0, 4000, 4000)
    fv = (m / (2 * np.pi * T * k)) ** (3 / 2) * np.exp(-m * v ** 2 / (2 * T * k)) * 4 * np.pi * v * v
    hist.plot(v, fv, label="Maxwell–Boltzmann distribution")
    hist.legend(loc="upper right")

    # Maxwell-Boltsman velocity[i] distribution
    v_i = np.linspace(-3000, 3000, 6000)
    fv_i = np.sqrt(m / (2 * np.pi * k * T)) * np.exp(-m * v_i ** 2 / (2 * T * k))
    hist_x.plot(v_i, fv_i, label="Maxwell–Boltzmann distribution")
    hist_x.legend(loc="upper right")
    hist_y.plot(v_i, fv_i, label="Maxwell–Boltzmann distribution")
    hist_y.legend(loc="upper right")
    hist_z.plot(v_i, fv_i, label="Maxwell–Boltzmann distribution")
    hist_z.legend(loc="upper right")

    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    slider = Slider(slider_ax,  # the axes object containing the slider
                    't',  # the name of the slider parameter
                    0,  # minimal value of the parameter
                    tfin,  # maximal value of the parameter
                    valinit=0,  # initial value of the parameter
                    color='#5c05ff'
                    )

    def update(time):
        i = int(np.rint(time / timestep))

        # ax.set_title('Energy =' + str(Energy[i]))

        # Draw Particles as circles
        # for j in range(particle_number):
        #     circle[j].center = particle_list[j].solpos[i][0], particle_list[j].solpos[i][1]
        hist.clear()
        hist_x.clear()
        hist_y.clear()
        hist_z.clear()

        vel_x = np.array([])
        vel_y = np.array([])
        vel_z = np.array([])
        vel_mod = np.array([])
        for j in range(len(particle_list)):
            if 3 <= particle_list[j].solpos[0][2] <= 4:
                vel_x = np.append(vel_x, particle_list[j].solvel[i][0])
                vel_y = np.append(vel_y, particle_list[j].solvel[i][1])
                vel_z = np.append(vel_z, particle_list[j].solvel[i][2])
                vel_mod = np.append(vel_mod, particle_list[j].solvel_mag[i])

        # Graph Particles speed histogram
        #vel_mod = [particle_list[j].solvel_mag[i] for j in range(len(particle_list))]
        hist.hist(vel_mod, bins=20, density=True, label="Simulation Data")
        hist.set_xlabel("Speed")
        hist.set_ylabel("Frecuency Density")

        # Graph Particles velocity[i] histogram
        dir = 0
        let = 'x'
        #vel_x = [particle_list[j].solvel[i][dir] for j in range(len(particle_list))]
        hist_x.hist(vel_x, bins=20, density=True, label="Simulation Data")
        hist_x.set_xlabel("Vel_" + let)
        hist_x.set_ylabel("Frecuency Density")

        dir = 1
        let = 'y'
        #vel_y = [particle_list[j].solvel[i][dir] for j in range(len(particle_list))]
        hist_y.hist(vel_y, bins=20, density=True, label="Simulation Data")
        hist_y.set_xlabel("Vel_" + let)
        hist_y.set_ylabel("Frecuency Density")

        dir = 2
        let = 'z'
        #vel_z = [particle_list[j].solvel[i][dir] for j in range(len(particle_list))]
        hist_z.hist(vel_z, bins=20, density=True, label="Simulation Data")
        hist_z.set_xlabel("Vel_" + let)
        hist_z.set_ylabel("Frecuency Density")

        # Compute 2d Boltzmann distribution
        E = total_Energy(particle_list, i)
        Average_E = E / len(particle_list)
        k = 1.38064852e-23
        T = 2 * Average_E / (2 * k)
        m = particle_list[0].mass
        v = np.linspace(0, 4000, 4000)
        fv = (m / (2 * np.pi * T * k)) ** (3 / 2) * np.exp(-m * v ** 2 / (2 * T * k)) * 4 * np.pi * v * v
        hist.plot(v, fv, label="Maxwell–Boltzmann distribution")
        hist.legend(loc="upper right")
        # hist.set_xlim([0, 1500])
        # hist.set_ylim([0, 0.01])

        # Maxwell-Boltsman velocity[i] distribution
        v_i = np.linspace(-3000, 3000, 6000)
        fv_i = np.sqrt(m / (2 * np.pi * k * T)) * np.exp(-m * v_i ** 2 / (2 * T * k))
        hist_x.plot(v_i, fv_i, label="Maxwell–Boltzmann distribution")
        hist_x.legend(loc="upper right")
        hist_y.plot(v_i, fv_i, label="Maxwell–Boltzmann distribution")
        hist_y.legend(loc="upper right")
        hist_z.plot(v_i, fv_i, label="Maxwell–Boltzmann distribution")
        hist_z.legend(loc="upper right")

    slider.on_changed(update)
    plt.show()