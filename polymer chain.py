import numpy as np
import matplotlib.pyplot as plt

# Parameters
v = 500                # Given parameter v
delta_t_star = 0.001   # Non-dimensional time step
total_time_star = 10000  # Total time for the simulation
num_steps = int(total_time_star / delta_t_star)  # Number of time steps

# Initial conditions
r1_star = np.array([0.0, 0.0])  # Initial position of the first bead
r2_star = np.array([np.sqrt(v), 0.0])  # Initial position of the second bead

# Store the end-to-end distance for plotting
R_end_star = np.zeros(num_steps)

# Random noise
noise_strength = np.sqrt(6 * delta_t_star)

# Simulation loop
for t in range(num_steps):
    # Generate random noise
    n1 = np.random.normal(0, 1, 2)
    n2 = np.random.normal(0, 1, 2)
    
    # Calculate the vector R_star between the beads
    R_star = r2_star - r1_star
    R_star_mag = np.linalg.norm(R_star)
    
    # Update positions based on the equations of motion
    r1_star += noise_strength * n1 + delta_t_star * (-3 * R_star / (1 - R_star_mag**2) / R_star_mag)
    r2_star += noise_strength * n2 + delta_t_star * (3 * R_star / (1 - R_star_mag**2) / R_star_mag)
    
    # Store the magnitude of the end-to-end vector
    R_end_star[t] = np.linalg.norm(R_star)

# Calculate RMS value
R_end_rms = np.sqrt(np.mean(R_end_star ** 2))

# Plot the end-to-end distance as a function of time
time_star = np.arange(0, total_time_star, delta_t_star)
plt.plot(time_star, R_end_star, label="End-to-End Distance", color='blue')

# Plot the RMS value as a horizontal line
plt.axhline(y=R_end_rms, color='red', linestyle='-', linewidth=2, label=f'RMS Value: {R_end_rms:.2f}')

plt.xlabel('Non-dimensional Time t*')
plt.ylabel('End-to-End Distance R*')
plt.title('Brownian Dynamics Simulation of a Polymer Chain')
plt.legend()
plt.show()
