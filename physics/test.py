from vpython import *

scene1 = canvas(title="P69: Cube of Spheres")

L = 3  # Половина длины стороны (6/2)
R_sphere = 0.5

for x in [-L, L]:
    for y in [-L, L]:
        for z in [-L, L]:
            # Выбираем цвет в зависимости от координат, чтобы было "минимум два цвета"
            current_color = color.orange if (x + y + z) > 0 else color.red
            sphere(pos=vector(x, y, z), radius=R_sphere, color=current_color)

# (b) Добавляем стрелку из одного угла в противоположный
arrow(pos=vector(-L, -L, -L), axis=vector(2*L, 2*L, 2*L), color=color.green, shaftwidth=0.2)