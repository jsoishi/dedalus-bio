from scipy.special import erf
import numpy as np

def sheet(x,y,k,omega,t,delta,hw,b,center=(0, 0)):
    x -= center[0]
    y -= center[1]
    yy_pos = y + b*np.sin(k*x - omega*t)
    pos = 0.5*(erf((yy_pos+hw)/delta) - erf((yy_pos-hw)/delta))


    x_vel = omega*(
				- (1/32 * b**4) 
				+ ((1/4 * b**2) - (1/8 * b**4))*np.cos(2*(k*x - omega*t))
				- (3/64)*b**4*np.cos(4*(k*x - omega*t))
				- 1
    	  	)

    y_vel = omega*(
		  		- (b - (1/8)*b**3)*np.cos(k*x - omega*t)
		  		- ((1/8)*b**3)*np.cos(3*(k*x - omega*t))
	  		)


    return pos, x_vel, y_vel
