from vpython import *

scene3 = canvas(title="P79: Bouncing Particle")

# Создаем полупрозрачную стенку (box)
wall = box(pos=vector(0,0,-1), size=vector(5, 5, 0.5), color=color.red, opacity=0.4)

# Создаем частицу
particle = sphere(pos=vector(-5, 0, -5), radius=0.3, color=color.cyan, make_trail=True)

# Начальная скорость
v = vector(0.5, 0, 0.5)
delta_t = 0.05

while True:
    rate(100) # Скорость анимации
    
    # Двигаем частицу
    particle.pos = particle.pos + v * delta_t
    
    # Условие отскока (P79): 
    # Если частица касается стены по оси Z
    if abs(particle.pos.z - wall.pos.z) < 0.3:
        v.z = -v.z  # Инвертируем скорость по Z