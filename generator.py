import numpy as np
import matplotlib.pyplot as plt

#program to generate the measurement file
N = 500
x = .4 + 4* np.random.normal(loc=0,scale=1,size=N)
a = .1;b = 1;c=2;
y = a+b*np.sin(x) + c*np.sin(x)**2 + .5*np.random.normal(loc=0,scale=1,size=N)
plt.plot(x,y,'r.')
plt.show()
np.save('means.npy',np.append(x.reshape(N,1),y.reshape(N,1)))
