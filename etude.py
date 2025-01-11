#######  np.linalg   #######
import numpy as np

# Define a square matrix
A = np.array([[4, 2], [3, 1]])

# Inverse of the matrix
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)

# Determinant of the matrix
det_A = np.linalg.det(A)
print("Determinant of A:", det_A)

# Solve the linear equation Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b) 
print("Solution of Ax = b:", x)

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)


#######  numpy.fft  #######

import matplotlib.pyplot as plt

# Create a sample signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# Compute the Fourier Transform
fft_signal = np.fft.fft(signal)

# Absolute value to get magnitudes
fft_abs = np.abs(fft_signal)

# Plot the frequency spectrum (magnitudes)
frequencies = np.fft.fftfreq(len(signal), d=t[1] - t[0])
plt.plot(frequencies, fft_abs)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 200)  # Limit to show relevant frequencies
plt.show()

#######   numpy.random  #######
# Set a random seed for reproducibility (optional)
np.random.seed(42)

# Generate random numbers
random_uniform = np.random.rand(3)  # Uniform distribution between 0 and 1
random_normal = np.random.randn(4)  # Standard normal distribution

# Generate random integers
random_integers = np.random.randint(1, 11, size=(2, 3))  # Between 1 (inclusive) and 10 (exclusive)

# Random choice from an array
choices = np.array(['heads', 'tails'])
random_choice = np.random.choice(choices, size=5)

print("Random uniform:", random_uniform)
print("Random normal:", random_normal)
print("Random integers:", random_integers)
print("Random choices:", random_choice)