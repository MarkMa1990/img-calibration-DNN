

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("mean_result.txt")

plt.figure(0)
plt.subplot(111)
plt.plot(data,'b-^')
plt.xlabel("Epoch")
plt.ylabel("RMS")

plt.savefig('fig')

plt.show()

