import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ABBIRB140:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        """Initialize ABB IRB 140 with CoppeliaSim handles."""
        # DH Parameters: [a, d, alpha]
        self.dh_params = [
            [0, 0.352, np.pi / 2],
            [0.070, 0, 0],
            [0.360, 0, np.pi / 2],
            [0.239, 0, -np.pi / 2],
            [0.141, 0, np.pi / 2],
            [0.065, 0, 0],
        ]

    def init_coppelia(self):
        self.robot_handle = self.sim.getObject('/IRB140')

        # Retrieve joint handles
        self.joint_handles = [
            self.sim.getObject(f"/IRB140/joint"),
            self.sim.getObject(f"/IRB140/link/joint"),
            self.sim.getObject(f"/IRB140/link/joint/link/joint"),
            self.sim.getObject(f"/IRB140/link/joint/link/joint/link/joint"),
            self.sim.getObject(f"/IRB140/link/joint/link/joint/link/joint/link/joint"),
            self.sim.getObject(f"/IRB140/link/joint/link/joint/link/joint/link/joint/link/joint"),
        ]

    def dh_transform(self, a, d, alpha, theta):
        """Compute a single DH transformation matrix."""
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ])

    def forward_kinematics(self, joint_angles):
        """Compute forward kinematics for given joint angles."""
        T = np.eye(4)
        for i, theta in enumerate(joint_angles):
            a, d, alpha = self.dh_params[i]
            T = np.dot(T, self.dh_transform(a, d, alpha, theta))
        return T

    def cubic_trajectory(self, p_start, p_end, v_start, v_end, t):
        """Generate cubic trajectory for a single joint."""
        T = t[-1]  # Total time
        a0 = p_start
        a1 = v_start
        a2 = (3 * (p_end - p_start) - (2 * v_start + v_end) * T) / (T**2)
        a3 = (-2 * (p_end - p_start) + (v_start + v_end) * T) / (T**3)

        positions = a0 + a1 * t + a2 * t**2 + a3 * t**3
        velocities = a1 + 2 * a2 * t + 3 * a3 * t**2
        accelerations = 2 * a2 + 6 * a3 * t

        return positions, velocities, accelerations

    def path_planning(self, joint_start, joint_end, total_time):
        """Generate joint space trajectories between two configurations."""
        t = np.linspace(0, total_time, 100)  # Time steps
        trajectories = []

        for i in range(len(joint_start)):
            positions, velocities, accelerations = self.cubic_trajectory(
                joint_start[i], joint_end[i], 0, 0, t
            )
            trajectories.append(positions)

        return np.array(trajectories)
        

    # def simulate(self, joint_trajectories):
    #     """Simulate the robot motion in CoppeliaSim and visualize real-time data."""
    #     joint_positions = [[] for _ in range(len(self.joint_handles))]
    #     joint_velocities = [[] for _ in range(len(self.joint_handles))]
    #     timestamps = []

    #     # Initialize live plot
    #     fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    #     lines_pos = [ax[0].plot([], [], label=f"Joint {i+1}")[0] for i in range(len(self.joint_handles))]
    #     lines_vel = [ax[1].plot([], [], label=f"Joint {i+1}")[0] for i in range(len(self.joint_handles))]

    #     ax[0].set_title("Joint Positions (radians)")
    #     ax[0].set_xlim(0, len(joint_trajectories.T) * 0.05)  # Assuming a 50ms delay
    #     ax[0].set_ylim(-np.pi, np.pi)
    #     ax[0].legend()

    #     ax[1].set_title("Joint Velocities (rad/s)")
    #     ax[1].set_xlim(0, len(joint_trajectories.T) * 0.05)
    #     ax[1].set_ylim(-2, 2)
    #     ax[1].legend()

    #     def update_plot(frame):
    #         for j, joint_angle in enumerate(joint_trajectories[:, frame]):
    #             self.sim.setJointTargetPosition(self.joint_handles[j], joint_angle)

    #         time.sleep(0.05)  # Simulation delay

    #         # Ensure timestamps length matches the frames processed
    #         if len(timestamps) <= frame:
    #             timestamps.append(frame * 0.05)

    #         for i, joint_handle in enumerate(self.joint_handles):
    #             # Get joint position and velocity
    #             position = self.sim.getJointPosition(joint_handle)
    #             velocity = self.sim.getObjectVelocity(joint_handle)[1]

    #             # Append position and velocity
    #             if len(joint_positions[i]) <= frame:
    #                 joint_positions[i].append(position)
    #             if len(joint_velocities[i]) <= frame:
    #                 joint_velocities[i].append(velocity)

    #             # Synchronize timestamps and data lengths
    #             current_length = len(joint_positions[i])
    #             if len(timestamps) > current_length:
    #                 trimmed_timestamps = timestamps[:current_length]
    #             else:
    #                 trimmed_timestamps = timestamps

    #             # Update the plot data
    #             lines_pos[i].set_data(trimmed_timestamps, joint_positions[i])
    #             lines_vel[i].set_data(trimmed_timestamps, joint_velocities[i])

    #         # Adjust axis limits
    #         ax[0].relim()
    #         ax[0].autoscale_view()
    #         ax[1].relim()
    #         ax[1].autoscale_view()

    #         return lines_pos + lines_vel

    #     ani = FuncAnimation(fig, update_plot, frames=len(joint_trajectories.T), repeat=False)
    #     plt.show()

    def simulate(self, joint_trajectories):
        """Simulate the robot motion in CoppeliaSim without visualization."""
        # Loop through each time step in the trajectory
        for frame in range(joint_trajectories.shape[1]):
            for j, joint_angle in enumerate(joint_trajectories[:, frame]):
                # Set the target position of each joint
                self.sim.setJointTargetPosition(self.joint_handles[j], joint_angle)
            
            # Wait for a small duration to simulate real-time execution
            time.sleep(0.05)  # Adjust delay for smoother simulation

    def run_coppelia(self):
        # Initialize connection to CoppeliaSim
        self.sim.startSimulation()

        # Define start and end joint configurations
        joint_start = [0, 0, 0, 0, 0, 0]  # Radians
        joint_end = [np.pi / 3, -np.pi / 6, np.pi / 4, -np.pi / 4, np.pi / 3, - np.pi / 3]

        # Path planning
        total_time = 5  # Total time for trajectory (seconds)
        for _ in range(1):
            joint_trajectories = self.path_planning(joint_start, joint_end, total_time)

            # Simulate motion and visualize real-time graphs
            print('yes')
            self.simulate(joint_trajectories)

        # Stop simulation
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = ABBIRB140()
    controller.init_coppelia()
    controller.run_coppelia()
