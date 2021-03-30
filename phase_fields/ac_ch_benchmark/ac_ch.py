"""ac_ch.py

Coupled Allen-Cahn Cahn-Hilliard equations for phase fields.

Test problem is problem 1 from PRISMS-PF paper,
DeWitt, Rudraraju, Montiel, Andrews, Thornton (2020, NPJ Computational Materials)

"""
from mpi4py import MPI
import os
import time
import logging
import numpy as np

import dedalus.public as de

from dedalus.tools  import post
from dedalus.extras import flow_tools
import pathlib

start = time.time()
logger = logging.getLogger(__name__)


# parameters
data_dir = pathlib.Path('data/ac_ch')
if MPI.COMM_WORLD.Get_rank() == 0:
    data_dir.mkdir(parents=True, exist_ok=True)
mesh = [4,4]
nx = 16#128
ny = 16#128
nz = 16#128
ε = 0.01

L_box = 100

M = 1
L = 100
κ = 2

x = de.Fourier('x', nx, interval=(0,L_box))
y = de.Fourier('y', ny, interval=(0,L_box))
z = de.Fourier('z', nz, interval=(0,L_box))

domain = de.Domain([x, y, z], grid_dtype='float', mesh=mesh)

problem = de.IVP(domain, variables=['η', 'c'])

problem.parameters['L'] = L
problem.parameters['κ'] = κ
problem.parameters['M'] = M

problem.substitutions["Lap(A)"] = "dx(dx(A)) + dy(dy(A)) + dz(dz(A))"
problem.substitutions["h"] = "3*η**2 - 2*η**3"
problem.substitutions["fα"] = "2*c**2"
problem.substitutions["fβ"] = "2*c**2 - 4*c + 2"
problem.substitutions["dh"] = "6*η - 6*η**2"
problem.substitutions["dfα"] = "4*c"
problem.substitutions["dfβ"] = "4*c - 4"

# Allen-Cahn
problem.add_equation("dt(η) = -L*((fβ - fα)*dh - κ*Lap(η))")

# Cahn-Hilliard
problem.add_equation("dt(c) = - M*Lap(dfα*(1-h) + dfβ*h)")

solver = problem.build_solver(de.timesteppers.RK443)

analysis_tasks = []
snap = solver.evaluator.add_file_handler(data_dir/pathlib.Path('snapshots'), iter=100, max_writes=200)
snap.add_task("c")
snap.add_task("η")
analysis_tasks.append(snap)

# Create initial conditions
def ball(x, x0, r, eps):
    rr = np.sqrt((x[0] - x0[0])**2 + (x[1] - x0[1])**2 + (x[2] - x0[2])**2)
    
    # 0 <= f <= 1 
    f = (1-np.tanh((rr - r)/(np.sqrt(2)* eps)))/2

    return f    

c0 = solver.state['c']
η0 = solver.state['η']
x0 = [40, 40, 40]
x1 = [65, 65, 65]
r0 = 20
r1 = 12

cmax = 1
cmin = 0.4

c0['g'] = (cmax-cmin)* (ball(domain.grids(), x0, r0,0.1) + ball(domain.grids(), x1, r1,0.1)) + cmin
η0['g'] = ball(domain.grids(), x0, r0, ε) + ball(domain.grids(), x1, r1, ε)

solver.stop_wall_time = 24*3600
solver.stop_iteration = 10#000
solver.stop_sim_time = 5
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("integ(c)", name="Cint")
flow.add_property("integ(η)", name="ηint")

dt = 1e-3
start_run_time  = time.time()
try:
    while solver.ok:
        if (solver.iteration-1) % 10 == 0:
            logger.info("Step {:d}".format(solver.iteration))
            logger.info("Integrated c = {:10.7e}".format(flow.max("Cint")))
            logger.info("Integrated η = {:10.7e}".format(flow.max("ηint")))
        solver.step(dt)
except:
    logger.error('Exception raised, triggering end of main loop.')
finally:
    solver.evaluate_handlers_now(dt)
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %f' %(end_run_time-start_run_time))

stop = time.time()
logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
