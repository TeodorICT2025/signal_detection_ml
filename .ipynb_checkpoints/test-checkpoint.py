import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
xx = np.sin(2*np.pi*12*t)


plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Test signal")

plt.show()   # 👈 ASTA E CHEIA
# FFT
X = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), d=t[1]-t[0])

plt.figure()
plt.plot(freq, np.abs(X))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("Frequency domain")
plt.xlim(0, 20)   # vezi clar vârful
plt.show()

