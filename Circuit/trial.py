import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 1.75e-3, 100, dtype=float)

eng = matlab.engine.start_matlab()
p1, p2, p3 = eng.analytical_solution(t, nargout=3)

plt.figure(1)
plt.plot(t,np.array(p1))
plt.plot(t,np.array(p2))
plt.plot(t,np.array(p3))
plt.show()