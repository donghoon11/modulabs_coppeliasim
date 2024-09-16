import numpy as np
import matplotlib.pyplot as plt

class KF():
    def __init__(self):
        # Initialization for system model.
        self.A = 1
        self.H = 1
        self.Q = 0
        self.R = 4
        
        # Initialization for estimation.
        self.x_0 = 12  # 14 for book.
        self.P_0 = 6
        
        # Input parameters.
        self.time_end = 10
        self.dt = 0.2

    def get_volt(self):
        """Measure voltage."""
        v = np.random.normal(0, 2)   # v: measurement noise.
        volt_true = 14.4             # volt_true: True voltage [V].
        z_volt_meas = volt_true + v  # z_volt_meas: Measured Voltage [V] (observable).
        
        return z_volt_meas

    def kalman_filter(self, z_meas, x_esti, P):
        """Kalman Filter Algorithm for One Variable."""
        # (1) Prediction.
        x_pred = self.A * x_esti
        P_pred = self.A * P * self.A + self.Q

        # (2) Kalman Gain.
        K = P_pred * self.H / (self.H * P_pred * self.H + self.R)

        # (3) Estimation.
        x_esti = x_pred + K * (z_meas - self.H * x_pred)

        # (4) Error Covariance.
        P = P_pred - K * self.H * P_pred

        return x_esti, P
    
    def run(self):
        time = np.arange(0, self.time_end, self.dt)
        n_samples = len(time)
        volt_meas_save = np.zeros(n_samples)
        volt_esti_save = np.zeros(n_samples)
        x_esti, P = None, None
        for i in range(n_samples):
            z_meas = self.get_volt()
            if i == 0:
                x_esti, P = self.x_0, self.P_0
            else:
                x_esti, P = self.kalman_filter(z_meas, x_esti, P)

            volt_meas_save[i] = z_meas
            volt_esti_save[i] = x_esti

        plt.plot(time, volt_meas_save, 'r*--', label='Measurements')
        plt.plot(time, volt_esti_save, 'bo-', label='Kalman Filter')
        plt.legend(loc='upper left')
        plt.title('Measurements v.s. Estimation (Kalman Filter)')
        plt.xlabel('Time [sec]')
        plt.ylabel('Voltage [V]')
        plt.savefig('/home/oh/my_coppeliasim/modulabs_coppeliasim/localization/png/simple_kalman_filter.png')

if __name__ == "__main__":
    kalman = KF()
    kalman.run()