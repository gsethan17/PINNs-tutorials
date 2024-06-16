import numpy as np
import seaborn as sns
from scipy.integrate import solve_ivp

class PhysicsModel(object):
    def __init__(self, name="None"):
        self.name = name

class Projectile2D(PhysicsModel):
    def __init__(self, name="Projectile2D"):
        super().__init__(name)
        self.num_u = 2
        self.num_s = 2
        self.num_z = 1
        
        self.min_u = [0., 0.]
        self.min_u = [5., 5.]
        
        self.mu = 0.05
        
    def simulate(self, u0, t_max=1.):
        t_eval = np.arange(0.001, t_max+0.001, 0.001)
        t_span = [min(t_eval), max(t_eval)]
        
        init_values = [0, 0] + u0
        
        def get_solution(t, s):
            ax = (-1*self.mu) * np.sqrt(s[2]**2+s[3]**2) * s[2]
            ay = (-1*self.mu) * np.sqrt(s[2]**2+s[3]**2) * s[3] - 9.81
            
            output_ = [s[2], s[3], ax, ay]
            return output_
        
        def solve_pde(t, s):
            ax = 0
            ay = -9.81
            
            output_ = [s[2], s[3], ax, ay]
            return output_
        
        solution = solve_ivp(get_solution,
                            y0=init_values,
                            t_span=t_span,
                            t_eval=t_eval,
                            )
        s = solution.y[:2, :]
        s[np.where(s<0)] = 0.
        
        numerical = solve_ivp(solve_pde,
                            y0=init_values,
                            t_span=t_span,
                            t_eval=t_eval,
                            )
        s_num = numerical.y[:2, :]
        s_num[np.where(s_num<0)] = 0.
        
        return s, s_num
    
    def measure(self, u0, t_max=1.):
        t_eval = np.arange(0.001, t_max+0.001, 0.001)
        t_span = [min(t_eval), max(t_eval)]
        
        mu = np.random.normal(loc=self.mu, scale=0.001)
        init_values = [0, 0] + u0
        
        def get_solution(t, s):
            ax = (-1*mu) * np.sqrt(s[2]**2+s[3]**2) * s[2]
            ay = (-1*mu) * np.sqrt(s[2]**2+s[3]**2) * s[3] - 9.81
            
            output_ = [s[2], s[3], ax, ay]
            return output_
        
        solution = solve_ivp(get_solution,
                            y0=init_values,
                            t_span=t_span,
                            t_eval=t_eval,
                            )
        s = solution.y[:2, :]
        s[np.where(s<0)] = 0.
        
        if len(np.where(s[1, 1:]==0)[0]) > 0:
            end_t = np.where(s[1, 1:]==0)[0][0]
        else:
            end_t = len(s[1])
            
        t_meas = np.random.normal(loc=(end_t * 0.001) + 0.001, scale=0.001)
        
        x_meas = np.random.normal(loc=s[0, end_t], scale=0.05)
        
        return t_meas, x_meas
    
    def show_measure(self, xs):
        sns.kdeplot(xs, linewidth=2.5, label='Distrubution of Measured X Positions')
        plt.xlim([-0.2, 5.2])
        plt.grid()
        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Density")
        plt.show()

    def show_simulate(self, u0, 
                      data=np.array([]),
                      show=True,
                      ):
        s, s_num = self.simulate(u0)
        if len(np.where(s[1, 1:]==0)[0]) > 0:
            end_t = np.where(s[1, 1:]==0)[0][0]
        else:
            end_t = len(s[1])
        
        plt.arrow(0, 0, u0[0]*0.1, u0[1]*0.1, ec='k', fc='c', width=0.02, head_width=0.1, shape='right', label='Input')
        plt.plot(s[0, :end_t+2], s[1, :end_t+2], c='gray', linewidth=2, label='GT')
        if len(data) > 0:
            plt.scatter(data, [0 for i in range(len(data))], s=50, facecolors="None", edgecolors='r', label='DATA')
        plt.plot(s_num[0, :end_t+2], s_num[1, :end_t+2], c='k', linestyle='--', label='RK45')
        if show:
            self.show()
            
    def show_networks(self, u0, networks, 
                      ts=np.arange(0., 1+0.1, 0.1),
                      data=np.array([]),
                      show=True,
                      ):
        self.show_simulate(u0, data=data, show=False)
        
        inputs = []
        for t in ts:
            inputs.append(u0 + [t])
        outputs = networks(np.array(inputs))
            
        plt.scatter(outputs[..., :1], outputs[..., 1:], label='PINNs')
        if show:
            self.show()
    
    def show(self):
        plt.grid()
        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        

import matplotlib.pyplot as plt 
if __name__ == "__main__":
    model = Projectile2D()
    model.measure([5., 5.])