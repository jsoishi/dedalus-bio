"""
Dedalus script for 2D incompressible hydrodynamics with moving immersed boundary.

This script uses a Fourier basis in both y directions.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run using 4 processes (e.g), you could use:
    $ mpiexec -n 4 python3 swimming_sheet.py


"""

import numpy as np
from mpi4py import MPI
import time
import body as bdy

from dedalus import public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.rank

# Parameters
dt = 0.001 # timestep
Lx = 8.    # box size in x
Ly = 8.    # box size in y
ν  = 0.1   # viscosity
γ  = 40    # coupling parameter for immersed boundary
hw = 0.1   # half-width of sheet
b = 0.1    # amplitude of sheet (b in Taylor 1951 eq 1)
sigma = 1. # swimming frequency (sigma in Taylor 1951 eq 1)
k = 4*np.pi/Ly # wavenumber of sheet frequency (k in Taylor 1951 eq 1)
delta = 0.05 # tanh width for mask;

# Initial body parameters
x0,U0 = 0,0
y0,V0 = 0,0

# Create bases and domain
x_basis = de.Fourier('x',384, interval=(-Lx, Lx), dealias=3/2)
y_basis = de.Fourier('y',384, interval=(-Ly, Ly), dealias=3/2)
domain  = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Shift up y function for field
def shift_field_y(field, shift_y):
    return np.concatenate((field[shift_y:], field[:shift_y]), axis=0)
    
    
# Setup mask function
x,y = domain.grids(scales=domain.dealias)
K = domain.new_field()
V = domain.new_field()
J = domain.new_field()
W = domain.new_field()
K.set_scales(domain.dealias,keep_data=False)
V.set_scales(domain.dealias,keep_data=False)
J.set_scales(domain.dealias,keep_data=False)
W.set_scales(domain.dealias,keep_data=False)
K['g'], V['g'] =  bdy.sheet(x, y, k, sigma, 0, delta, hw, b)
# Make fields J, W which are identical to K, V but shifter up in the y axis
J['g'] = shift_field_y(K['g'], 40)
W['g'] = shift_field_y(V['g'], 40)

# 2D Incompressible hydrodynamics
problem = de.IVP(domain, variables=['p','u','v','ωz'])

problem.parameters['ν']   = ν
problem.parameters['γ']   = γ
problem.parameters['K']   = K
problem.parameters['V']   = V
problem.parameters['J']   = J
problem.parameters['W']   = W
problem.add_equation("dt(u) + ν*dy(ωz) + dx(p) =  ωz*v -γ*(K + J)*u") # I'm not so sure about adding J into this equation
problem.add_equation("dt(v) - ν*dx(ωz) + dy(p) = -ωz*u -γ*K*(v-V) - γ*J*(v-W)") # or this one
problem.add_equation("ωz + dy(u) - dx(v) = 0")
problem.add_equation("dx(u) + dy(v) = 0",condition="(nx != 0) or (ny != 0)")
problem.add_equation("p = 0",condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
ωz = solver.state['ωz']
u = solver.state['u']

# Integration parameters
t_wave = 2*np.pi/sigma
solver.stop_sim_time = 2*t_wave #np.inf
solver.stop_wall_time = 10*60*60.
solver.stop_iteration = 2000

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_test', iter=20, max_writes=50)
snapshots.add_task("p")
snapshots.add_task("u")
snapshots.add_task("v")
snapshots.add_task("ωz")
snapshots.add_task("K")
snapshots.add_task("V")
snapshots.add_task("J")
snapshots.add_task("W")

# Runtime monitoring properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=20)
flow.add_property("abs(ωz)", name='q')

analysis_tasks = []
analysis_tasks.append(snapshots)

# Tasks for force computation
# force = flow_tools.GlobalFlowProperty(solver, cadence=1)
# force.add_property("K*(v-V)", name='G0')
# force.add_property("-y*K*(u-U)+x*K*(v-V)", name='T0')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        #G0 = γ*force.volume_average('G0')
        #τ0 = γ*force.volume_average('T0') + y0*F0 - x0*G0
        #F0,G0,τ0 = F0/μ, G0/μ - gravity, τ0/(μ*I)
        #y0 = y0 + V0*dt
        #V0 = V0 + G0*dt
        K['g'],V['g'] =  bdy.sheet(x, y, k, sigma, solver.sim_time, delta, hw, b)
        J['g'] = shift_field_y(K['g'], 40)
        W['g'] = shift_field_y(V['g'], 40)
        
        if (solver.iteration-1) % 20 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max ωz = %f' %flow.max('q'))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)




