from scipy.special import erf
import numpy as np

def sheet(x,y,k,omega,t,delta,hw,ampl,center=(0, 0)):
    x -= center[0]
    y -= center[1]
    yy_pos = y + ampl*np.sin(k*x - omega*t)
    pos = 0.5*(erf((yy_pos+hw)/delta) - erf((yy_pos-hw)/delta))

    vel = - ampl*omega*np.cos(k*x - omega*t)
    return pos, vel
