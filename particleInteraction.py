import numpy as np

# Создаем класс "Particle" для хранения информации о каждой частице из модели (координаты, скорость, ...)
# Класс "Particle" содержит функцию compute_step для вычисления координат и скорости частицы для следующего шага с
# помощью метода молекулярной динамики. Также содержит функции вычисления скорости после столкновения.
class Particle:
    def __init__(self, mass, radius, epsilon, sigma, position, velocity, force, acceleration, alpha, adsorbate):
        self.mass = mass #масса частицы
        self.radius = radius #радиус частицы
        self.epsilon = epsilon #глубина потенциальной ямы
        self.sigma = sigma #расстояние, на котором энергия взаимодействия становится нулевой
        self.alpha = alpha

        self.adsorbate = adsorbate  #показывает, чем является частица - газ или твердое тело

        # позиция частицы, скорость, ускорение, энергия взаимодействия для данной итерации
        self.position = np.array(position)
        self.heat_vel = velocity
        self.flow_vel = np.array([0., 0., 0.])
        self.velocity = np.array(self.heat_vel + self.flow_vel)
        self.force = np.array(force)
        self.acceleration = np.array(acceleration)

        # все позиции частицы, скорости, модули скорости, полученные в ходе симуляции
        self.solpos = [np.copy(self.position)]
        self.solvel = [np.copy(self.velocity)]
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocity))]

    def compute_step(self, step, f, z, flow, borders):
        # Вычисляем позицию и скорость частицы для следующего шага
        # Через потенциалы вычисляем энергию взаимодействия атомов. Из второго закона ньютона находим ускорение частицы
        # Вычисляем новые координаты частицы методом верле
        a = self.acceleration
        self.force = f
        self.acceleration = self.force / (self.mass * 1e26)
        self.position += step * self.velocity + (1 / 2) * self.acceleration * step * step
        self.heat_vel += 1 / 2 * (a + self.acceleration) * step

        if self.adsorbate:
            self.flow_vel = np.array([flow[0], flow[1], flow[2]])

        self.velocity = self.heat_vel + self.flow_vel

        #Сохраняем скорость, позицию, модуль скорости в массив данных о частице для каждого шага
        self.solpos.append(np.copy(self.position))
        self.solvel.append(np.copy(self.velocity))
        self.solvel_mag.append(np.linalg.norm(np.copy(self.velocity)))

    def check_coll(self, particle):
        # проверяем столкновение частиц
        r1, r2 = self.radius, particle.radius
        x1, x2 = self.position, particle.position
        di = x2 - x1
        norm = np.linalg.norm(di)
        if norm - (r1 + r2) * 1.1 < 0:
            return True
        else:
            return False

    def compute_coll(self, particle, step, center, R, Z):
        # вычисляем скорость частиц после столкновения
        m1, m2 = (self.mass), (particle.mass)
        r1, r2 = (self.radius), (particle.radius)
        v1, v2 = self.heat_vel, particle.heat_vel
        x1, x2 = self.position, particle.position

        ads1, ads2 = self.adsorbate, particle.adsorbate
        if ads1 == 2:
            ads1 = 1
        if ads2 == 2:
            ads2 = 1
        di = x2 - x1
        norm = np.linalg.norm(di)
        if (norm) - (r1 + r2) * (1.1) < step * (abs(np.dot(v1 - v2, di)) / norm):
            if ads1 == ads2 and ads1 == 0:
                self.heat_vel = np.array([0., 0., 0.])
                particle.heat_vel = np.array([0., 0., 0.])
            # столкновение частиц газа друг с другом, вызов функции collide_particle
            elif ads1 == ads2 and ads1 == 1:
                collide_particle(self, particle, 0, step)
            # столкновение частиц газа с поверхностью, вызов функции collide_particle
            elif ads1:
                collide_particle(self, particle, 1, step)
                rght = center[0] + R
                lft = center[0] - R
                up = center[1] + R
                dwn = center[1] - R
                if (lft < particle.position[0] < rght and dwn < particle.position[1] < up and particle.position[2] < Z):
                    self.adsorbate = 2

    def compute_refl(self, step, size):
        # Вычисляем скорость частицы после столкновения с границей ячейки
        # Рассмотриваем систему объема V, состоящую из N частиц
        # Она находится в контакте с термостатом. Термостат поддерживает постоянную температуру T стенок системы.
        # Частица при ударе о стенку прилипает к ней, приобретает температуру стенки, а затем отлетает от стенки в произвольном направлении.
        # Модуль вектора скорости (v) отлетающей частицы уловлетворяет условию mv^2 / 2 = 3 / 2 * kT

        BoltsmanConstant = 1.38 * 10e-23 # k
        temperature = 300  # T, Кельвины
        velmod = 3 * BoltsmanConstant * temperature / self.mass

        r, v, x, h = self.radius, self.velocity, self.position, self.heat_vel
        projs = np.array([abs(v[0]), abs(v[1]), abs(v[2])])
        projx = step * (projs[0])
        projy = step * (projs[1])
        projz = step * (projs[2])

        if ((abs(x[0])) - (r) < projx):
            self.heat_vel[0] = np.sqrt(abs(velmod - self.heat_vel[1] ** 2 - self.heat_vel[2] ** 2))
        elif abs((size[0]) - (x[0])) - (r) < projx:
            self.heat_vel[0] = h[0]
            self.position[0] = self.position[0] - size[0]

        if abs((x[1])) - (r) < projy:
            self.heat_vel[1] = np.sqrt(abs(velmod - self.heat_vel[0] ** 2 - self.heat_vel[2] ** 2))
        elif abs((size[1]) - (x[1])) - (r) < projy:
            self.heat_vel[1] = -1 * np.sqrt(abs(velmod - self.heat_vel[0] ** 2 - self.heat_vel[2] ** 2))

        if abs((x[2])) - (r) < projz:
            self.heat_vel[2] = np.sqrt(abs(velmod - self.heat_vel[0] ** 2 - self.heat_vel[1] ** 2))
        elif abs((size[2]) - (x[2])) - (r) < projz:
            self.heat_vel[2] = -1 * np.sqrt(abs(velmod - self.heat_vel[0] ** 2 - self.heat_vel[1] ** 2))

        a = np.sum(np.square(self.heat_vel))
        if a != velmod:
            self.heat_vel *= np.sqrt(velmod/a)

        self.velocity[0] = (self.heat_vel[0]) + (self.flow_vel[0])
        self.velocity[1] = (self.heat_vel[1]) + (self.flow_vel[1])
        self.velocity[2] = (self.heat_vel[2]) + (self.flow_vel[2])

def collide_particle(particle1, particle2, type, step):
    #gas-gas
    if type == 0:
        sigma = particle1.sigma
        epsilon = particle1.epsilon
        x1, x2 = particle1.position, particle2.position
        m1, m2 = particle1.mass, particle2.mass
        r1, r2 = particle1.radius, particle2.radius
        v1, v2 = particle1.velocity, particle2.velocity
        r = np.linalg.norm(x2 - x1)

        if r < r1 + r2:
            r = r1 + r2

        # Рассчитываем энергию взаимодействия через потенциал Леннарда-Джонса
        if r < 2.5 * sigma:
            force = (48.) * (epsilon) * ((sigma ** 12) / ((r) ** 12) - (sigma ** 6) / ((r) ** 6))
        else:
            force = 0.

        # Рассматриваем столкнувшиеся частицы как единое тело, находим закон его движения vel
        U = - (x1 - x2) * force / r
        cnst = U * (m1 * 1e26 + m2 * 1e26) / (m1 * 1e26* m2 * 1e26)
        vel = cnst * step + cnst

        # Из закона движения находим скорости для каждой из столкнувшихся частиц отдельно
        particle1.heat_vel = (-m2 / (m1 + m2)) * (vel) * 100
        particle2.heat_vel = (m1 / (m1 + m2)) * (vel) * 100

        v1d = particle1.heat_vel + particle1.flow_vel
        v2d = particle2.heat_vel + particle2.flow_vel

        # Преобразование полученных скоростей, чтобы выполнялись законы сохранения импульса, энергии, массы
        sum = [m1 * v1[0] + m2 * v2[0], m1 * v1[1] + m2 * v2[1], m1 * v1[2] + m2 * v2[2]]
        for i in range (len(v1d)):
            v1d[i] += sum[i] / 2 / m1
            v2d[i] += sum[i] / 2 / m2

        particle1.velocity = v1d + particle1.flow_vel
        particle2.velocity = v2d + particle2.flow_vel

    #gas-surface
    else:
        sigma = np.sqrt(particle1.sigma*particle2.sigma)
        epsilon = np.sqrt(particle1.epsilon*particle2.epsilon)
        x1, x2 = particle1.position, particle2.position
        m1, m2 = particle1.mass, particle2.mass
        v1, v2 = particle1.velocity, particle2.velocity
        r1, r2 = particle1.radius, particle2.radius
        r = np.linalg.norm(x2 - x1)

        if r < r1 + r2:
            r = r1 + r2

        if r < 2.5 * sigma:
            force = (48.) * (epsilon) * ((sigma ** 12) / ((r) ** 12) - (sigma ** 6) / ((r) ** 6))
        else:
            force = 0.

        U = - (x1 - x2) * force / r
        cnst = U * (m1 * 1e26 + m2 * 1e26) / (m1 * 1e26 * m2 * 1e26)
        vel = cnst * step + cnst

        particle1.heat_vel = (-m2 / (m1 + m2)) * (vel) * 100.
        particle2.heat_vel = np.array([0., 0., 0])
        imaginaryvel = (m1 / (m1 + m2)) * (vel) * 100.

        v1d = particle1.heat_vel + particle1.flow_vel
        v2d = imaginaryvel + particle2.flow_vel

        # Преобразование полученных скоростей, чтобы выполнялись законы сохранения импульса, энергии, массы
        sum = [m1 * v1[0] + m2 * v2[0], m1 * v1[1] + m2 * v2[1], m1 * v1[2] + m2 * v2[2]]
        for i in range(len(v1d)):
            v1d[i] += sum[i] / 2 / m1
            v2d[i] += sum[i] / 2 / m2

        particle1.velocity = v1d + particle1.flow_vel
        particle2.velocity = particle2.heat_vel + particle2.flow_vel

# Вычисляем энергию парного взаимодействия с помощью потенциала Леннарда-Джонса
def LennardJones (particle_list, num):
    force_LJ = np.array([[(particle_list[i].force[0]), (particle_list[i].force[1]), (particle_list[i].force[2])]
                         for i in range (len(particle_list))])

    for i in range(num):
        for j in range(i + 1, len(particle_list)):

            if particle_list[i].adsorbate > 0 and particle_list[j].adsorbate > 0:
                sigma = particle_list[i].sigma
                epsilon = particle_list[i].epsilon
            else:
                sigma = np.sqrt(particle_list[i].sigma * particle_list[j].sigma)
                epsilon = np.sqrt(particle_list[i].epsilon * particle_list[j].epsilon)

            r = np.sqrt(np.sum(np.square(particle_list[i].position - particle_list[j].position)))

            if r < particle_list[i].radius + particle_list[j].radius:
                r = particle_list[i].radius + particle_list[j].radius

            if r < 2.5 * sigma:
                force = (48.) * (epsilon) * ((sigma ** 12) / ((r) ** 13) - (0.5) * (sigma ** 6) / ((r)** 7))
            else: force = 0.

            vel = -(particle_list[i].position - particle_list[j].position) * force / r

            force_LJ[i][0] += vel[0]
            force_LJ[i][1] += vel[1]
            force_LJ[i][2] += vel[2]
            force_LJ[j][0] -= vel[0]
            force_LJ[j][1] -= vel[1]
            force_LJ[j][2] -= vel[2]

    return force_LJ

# Вычисляем энергию парного взаимодействия с помощью потенциала Morze
def Morze (particle_list):
    force_M = np.array([[(0.), (0.), (0.)] for i in range (len(particle_list))])

    for i in range(len(particle_list)):
        for j in range(i + 1, len(particle_list)):
            r = (np.sqrt(np.sum(np.square(particle_list[i].position - particle_list[j].position))))
            force = (particle_list[i].epsilon) * (particle_list[i].alpha) * np.exp((particle_list[i].alpha)*((particle_list[i].sigma) - (r)))

            vel = -(particle_list[i].position - particle_list[j].position) * force / r

            force_M[i][0] += vel[0]
            force_M[i][1] += vel[1]
            force_M[i][2] += vel[2]
            force_M[j][0] -= vel[0]
            force_M[j][1] -= vel[1]
            force_M[j][2] -= vel[2]

    return force_M

########################################################################################################################
# Вычисляем позиции и скорости частиц для следующего шага
def solve_step(particle_list, Ar_num, Al_num, step, size, center, R, Z, flow):
    # 1. Проверяем столкновение с границей или другой частицей для каждой частицы
    for i in range(len(particle_list[:Ar_num])):
        particle_list[i].compute_refl(step, size)
        for j in range(i + 1, len(particle_list)):
            particle_list[i].compute_coll(particle_list[j], step, center, R, Z)

    # 2. С помощью метода молекулярной динамики вычисляем позицию и скорость
    # Предваррительно вычислив энергию взаимодействия всех частиц в системе
    force_M = Morze(particle_list[Ar_num:])
    force_LJ = LennardJones(particle_list, Ar_num)
    forces = np.concatenate([force_LJ, force_M])

    for i in range (Ar_num + Al_num):
        particle_list[i].compute_step(step, forces[i], Z, flow, size)