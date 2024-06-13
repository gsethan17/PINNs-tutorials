import numpy as np
import tensorflow as tf

class DataLoader(object):
    def __init__(self, physics_model):
        self.name = physics_model.name
        self.model = physics_model
        
    def get_random_u(self, batch_size):
        random_u = np.array([np.random.uniform(
                        low=self.model.min_u[i],
                        high=self.model.max_u[i],
                        size=batch_size,
                    ) for i in range(self.model.num_u)]).T
        
        return tf.convert_to_tensor(random_u, dtype=tf.float32)
        
    def get_random_input(self, batch_size):
        random_u = self.get_random_u(batch_size)
        random_t = tf.convert_to_tensor(np.random.uniform(
                low=0.0,
                high=1.0,
                size=batch_size
            ).reshape((batch_size, 1))
            , dtype=tf.float32)
        
        random_input = tf.concat([random_u, random_t], axis=-1)
        
        return random_input
        
    def get_data_points(self, size):
        ts = []
        xs = []
        for i in range(size):
            t_meas, x_meas = self.model.measure([5., 5.])
            ts.append(t_meas)
            xs.append(x_meas)
        
        ts = np.array(ts).reshape((size, 1))
        xs = np.array(xs).reshape((size, 1))
        
        return ts, xs
        
if __name__ == "__main__":
    from PDE import Projectile2D
    data_loader = DataLoader(physics_model=Projectile2D())
    data_loader.get_data_points(3)