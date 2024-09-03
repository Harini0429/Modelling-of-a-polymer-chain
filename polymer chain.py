import numpy as np
import matplotlib.pyplot as plt

def simulate_two_particles(T_s=10000, dt_s=0.001, nu=500, bk =10**-9):
    N = int(T_s / dt_s)
    
    # Initialize position vectors
    r1 = np.zeros((N, 3))
    r2 = np.zeros((N, 3))

    # Set initial positions
    r1[0] = [0, 0, 0]
    r2[0] = [np.sqrt(nu), 0, 0]

    # Precompute constants
   # sqrt6_dt_s = np.sqrt(6 * dt_s)
   # inv_nu_bk = 1 / (nu * bk)

    for i in range(1, N):
        
        R_star = r2[i-1] - r1[i-1]
        r_c = np.linalg.norm(R_star) * (1/(nu*bk))

        n1 = np.random.uniform(0, 1, 3)
        n2 = np.random.uniform(0, 1, 3)

        # Update positions
        r1[i] = r1[i-1] + np.sqrt(6 * dt_s) * n1 + ((3 - r_c**2) / (nu * (1 - r_c**2))) * R_star* dt_s
        r2[i] = r2[i-1] + np.sqrt(6 * dt_s) * n2 - ((3 - r_c**2) / (nu * (1 - r_c**2))) * R_star* dt_s

        # Compute updated R_v
        R_star = r2[i] - r1[i]
        r_c = np.linalg.norm(R_star) * (1/nu*bk)

    return r1, r2

def calculate_R_end(r1, r2, dt_interval=10):
    N = len(r1)
    R_end = np.zeros(N // dt_interval)
    times = np.zeros(N // dt_interval)

    for i in range(len(R_end)):
        index = (i + 1) * dt_interval - 1
        R_v = r2[index] - r1[index]
        R_end[i] = np.linalg.norm(R_v)
        times[i] = index * dt_s

    return times, R_end

# Parameters
T_s = 10000
dt_s = 0.001
dt_interval = 10


r1, r2 = simulate_two_particles(T_s, dt_s)

# Calculate R_end and its RMS value
times, R_end = calculate_R_end(r1, r2, dt_interval)
R_end_rms = np.sqrt(np.mean(R_end**2))

# Plot R_end vs. t_s
plt.figure(figsize=(10, 6))
plt.plot(times, R_end, 'b-', label='R_end')
plt.axhline(y=R_end_rms, color='k', linestyle='-', label=f'RMS = {R_end_rms:.4f}')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude of R_v')
plt.title('Magnitude of R_v vs Time')
plt.legend()
plt.grid(True)
plt.show()
