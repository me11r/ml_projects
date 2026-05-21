from vpython import * # Updated import

# ## SCENE CONFIGURATION ##
scene.width = 800
scene.height = 800

# ## CONSTANTS ##
mzofp = 1e-7  
qe = 1.6e-19
mproton = 1.7e-27
B0 = vector(0, 0, 0.2)  # Fixed: removed the 4th dimension
bscale = 1

# #### THIS CODE DRAWS A GRID ####
# #### AND DISPLAYS MAGNETIC FIELD ####
xmax = 0.4
dx = 0.1
yg = -0.1
x = -xmax

while x < xmax + dx:
    curve(pos=[vector(x, yg, -xmax), vector(x, yg, xmax)], color=color.gray(0.7))
    x = x + dx

z = -xmax
while z < xmax + dx:
    curve(pos=[vector(-xmax, yg, z), vector(xmax, yg, z)], color=color.gray(0.7))
    z = z + dx

x = -xmax
dx = 0.2
while x < xmax + dx:
    z = -xmax
    while z < xmax + dx:
        arrow(pos=vector(x, yg, z), axis=B0 * bscale, color=vector(0, 0.8, 0.8))
        z = z + dx
    x = x + dx

# #### OBJECTS AND INITIAL CONDITIONS ####
particle = sphere(
    pos=vector(0, 0.15, 0.3),
    radius=1e-2,
    color=color.yellow,
    make_trail=True
)

## make trail easier to see (thicker) ##
particle.trail_object.radius = particle.radius / 3

# Part (a) initial velocity (3 dimensions)
vparticle = vector(2e6, 0, 0)
p = mproton * vparticle
qparticle = qe
deltat = 5e-11
t = 0

# #######################################################
# ## SIMULATION LOOP ##
# #######################################################
while t < 3.34e-7:
    rate(500)
    
    # 1. Update velocity from current momentum
    vparticle = p / mproton
    
    # 2. Calculate the magnetic Lorentz force: F = q * (v x B)
    Fnet = qparticle * cross(vparticle, B0)
    
    # 3. Update momentum
    p = p + Fnet * deltat
    
    # 4. Update position
    particle.pos = particle.pos + (p / mproton) * deltat
    
    # 5. Advance time
    t = t + deltat