import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Toy signal: periodic stimulus at 0.2 Hz with phase shift
fs = 2.0  # Hz sampling
T = 200   # seconds
t = np.arange(0, T, 1/fs)
f_stim = 0.2  # Hz
phase = np.pi/4
amp = 1.0
signal = amp * np.cos(2*np.pi*f_stim*t - phase) + 0.4*np.random.randn(t.size)

# Fourier
yf = rfft(signal)
xf = rfftfreq(t.size, 1/fs)
amp_spec = 2.0/ t.size * np.abs(yf)
phase_spec = np.angle(yf)

# Find index closest to stimulus frequency
idx = np.argmin(np.abs(xf - f_stim))
stim_amp = amp_spec[idx]
stim_phase = phase_spec[idx]
print(f"Stimulus frequency component: amp={stim_amp:.3f}, phase={stim_phase:.3f} rad")

# Simple plots
plt.figure()
plt.plot(t, signal)
plt.xlabel("Time (s)"); plt.ylabel("a.u."); plt.title("Toy signal")
plt.tight_layout()
plt.savefig("figures/toy_signal.png", dpi=150)

plt.figure()
plt.plot(xf, amp_spec)
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude"); plt.title("Amplitude spectrum")
plt.tight_layout()
plt.savefig("figures/amp_spectrum.png", dpi=150)
