# Small script for drawing Gaussians with different variances

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 5))

sigma0 = 0.45
sigma1 = 0.85
sigma2 = 2

x = np.arange(-2, 2.01, 0.01)
y0 = 1/np.sqrt(2*np.pi*sigma0**2) * np.exp(-x**2 / (2*sigma0**2))
y1 = 1/np.sqrt(2*np.pi*sigma1**2) * np.exp(-x**2 / (2*sigma1**2))
y2 = 1/np.sqrt(2*np.pi*sigma2**2) * np.exp(-x**2 / (2*sigma2**2))

a0 = 1/np.sqrt(2*np.pi*sigma0**2) * np.exp(-1**2 / (2*sigma0**2))
a1 = 1/np.sqrt(2*np.pi*sigma1**2) * np.exp(-1**2 / (2*sigma1**2))
a2 = 1/np.sqrt(2*np.pi*sigma2**2) * np.exp(-1**2 / (2*sigma2**2))

plt.plot(x, y0, c="#ca7a83")
plt.plot(x, y1, c="#b90f22")
plt.plot(x, y2, c="#490a11")
ax.plot([1, 1], [0, np.max(y0)], ls="--", c=".5")
plt.xlabel(r"$x$")
plt.ylabel(r"$P(x)$")
plt.title("Low vs. High Gaussian Variance")
plt.legend(["Low variance", "Medium variance", "High variance", "Sample location"])
plt.gcf()
plt.savefig("output/plots/different_variances.pdf")
plt.close(fig)
