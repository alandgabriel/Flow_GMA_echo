#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:52:25 2021

@author: alan
"""
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

echoPath = '/home/alan/Documents/finalProject/EchoNet-Dynamic/Videos/'
vid_files = glob.glob(echoPath + '*.avi')
cap= cv2.VideoCapture(vid_files [0])  
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frames.append(frame)                    
cap.release()

cv2.imwrite('/home/alan/Documents/GMA/echo/f1.png', cv2.resize(frames [11],(500,500),interpolation= cv2.INTER_CUBIC))

cv2.imwrite('/home/alan/Documents/GMA/echo/f2.png', cv2.resize(frames [12],(500,500),interpolation= cv2.INTER_CUBIC))

#%%
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
from functools import reduce
import operator
import math
from scipy.interpolate import interp1d


def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy


def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1):

    if img.ndim == 3:
        img = img.mean(2)
 
    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
 
 
    for ii in range(niter):
 
        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
        
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
 
        # update the image
        imgout += gamma*(NS+EW)
 
 
    return imgout

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]


path = '/home/alan/Documents/finalProject/synthetic database/data/GE Vingmed Ultrasound/A4C/normal/'
gdt = scipy.io.loadmat ( path + 'ground_truth.mat' ) ['X_gt']
img = scipy.io.loadmat ( path + 'im_sim.mat' ) ['im_sim']
info = scipy.io.loadmat ( path + 'info.mat' ) ['info']
colors = 'rgbmyc'
segms =  (np.ones((6,1)) * np.arange(0,6)).flatten(order = 'F').reshape((-1,1)) * np.ones((1,5))
infoX = ((info[0][0][0]).T).reshape((-1,))
infoY = ((info[0][0][1]).T).reshape((-1,))

'''
for i in range (np.shape(img)[2]):
    
    plt.imshow(cv2.cvtColor(img[:, :, i], cv2.COLOR_RGB2BGR), aspect='auto', interpolation='none',extent=extents(infoX) + extents(infoY), origin='upper')
    xp = gdt[:,0,i].reshape((36,5), order = 'F')
    yp = (max (infoY) - gdt[:,2,i]).reshape((36,5), order = 'F')
    for k in range (6):
        plt.plot(xp [segms == k], yp [segms == k],'o', markerfacecolor = colors[k])
    plt.pause(0.1)
'''
#for i in range (np.shape(img)[2]-1):
i =14
x = gdt[:,0,i+1]
y = max (infoY) - gdt[:,2,i+1]
dx = x - gdt [:,0,i]
dy = gdt [:,2,i+1] - gdt [:,2,i]

norma = np.sqrt (dx**2 + dy**2)
plt.imshow(cv2.cvtColor(img[:, :, i+1], cv2.COLOR_RGB2BGR), aspect='auto', interpolation='none',extent=extents(infoX) + extents(infoY), origin='upper')
sc = plt.scatter(x, y, c=norma, cmap = plt.cm.jet)
plt.colorbar(sc)
plt.show()
#plt.figure()
#h =plt.hist2d(x, y,bins=10)
#plt.colorbar(h[3])
#plt.imshow()

I = img[:, :, 0]
plt.figure()
plt.imshow(I)
I = anisodiff(I, niter=40,kappa =30, option=1)
# Apply Wiener Filter
kernel = gaussian_kernel(5)
#I = wiener_filter(I, kernel, K = 10)
plt.figure()
plt.imshow(I)
'''

coords = tuple(zip(x, y)) 
center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
coordSorted = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
x = np.array( list(zip(*coordSorted))[0])
y = np.array( list(zip(*coordSorted))[1])
plt.figure()
plt.subplot(121)
plt.scatter(x,y)
plt.title('Puntos de contorno')

f = interp1d(np.arange(0,len(x)), x, kind='cubic')
x = f(np.linspace(0, len(x)-1,len(x)*5))
f = interp1d(np.arange(0,len(y)), y, kind='cubic')
y = f(np.linspace(0, len(y)-1,len(y)*5))
#ixumbral = ((x >= 0) * (x < frame.shape[1]-1)) * ((y >= 0) * (y < frame.shape[0]-1))
#x = x [ixumbral]
#y = y[ixumbral]
plt.figure()
plt.scatter(x,y,c='b')
plt.title('Puntos con interpolación cubica')
'''
#%%  INVERSE DISTANCE WEIGHTING


class Estimation():
    def __init__(self,datax,datay,dataz):
        self.x = datax
        self.y = datay
        self.v = dataz

    def estimate(self,x,y,using='ISD'):
        """
        Estimate point at coordinate x,y based on the input data for this
        class.
        """
        if using == 'ISD':
            return self._isd(x,y)

    def _isd(self,x,y):
        d = np.sqrt((x-self.x)**2+(y-self.y)**2)
        if d.min() > 0:
            v = np.sum(self.v*(1/d**2)/np.sum(1/d**2))
            return v
        else:
            return self.v[d.argmin()]
        
    
e = Estimation(x,y,norma)
zn = np.zeros((len(infoX),len(infoY)))
xn = np.linspace(0, max(infoX), len(infoX))
yn = np.linspace(0, max(infoY), len(infoY))

for i,xv in enumerate(xn):
    for j,yv in enumerate(yn):
        zn[i,j] = e.estimate(xv,yv)
plt.figure()        
sc = plt.imshow(zn.T,origin='lower', cmap = plt.cm.jet, extent = extents(xn) + extents(yn))
plt.colorbar(sc)
plt.show()



#%% KRIGING
from scipy.spatial.distance import *
from skgstat import *
'''

x = np.array([4.0, 2.0, 4.1, 0.3, 2.0])

y = np.array([5.5, 1.2, 3.7, 2.0, 2.5])

z = np.array([4.2, 6.1, 0.2, 0.7, 5.2])

s0 = [2., 2.]

distance_matrix = pdist([s0] + list(zip(x,y)))
'''

coords = list(zip(x, y)) 

V = Variogram(coords, norma, maxlag=90, n_lags=25, model='gaussian', normalize=False)


V.plot()

ok = OrdinaryKriging(V, mode='estimate')

xx, yy = np.mgrid[0:max(infoX):616j, 0:max(infoY):479j]

field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
s2 = ok.sigma.reshape(xx.shape)

plt.figure()        
sc = plt.imshow(field.T,origin='lower', cmap = plt.cm.jet, extent = extents(xn) + extents(yn))
plt.colorbar(sc)
plt.show() 

#%%

import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

OK = OrdinaryKriging(
    x,
    y,
    norma,
    variogram_model="gaussian",
    verbose=False,
    enable_plotting=False,
)

z, ss = OK.execute("grid", xn, yn)
    
plt.figure()        
sc = plt.imshow(z,origin='lower', cmap = plt.cm.jet, extent = extents(xn) + extents(yn))
plt.colorbar(sc)
plt.show() 

#%%
from sklearn.model_selection import RandomizedSearchCV
from pykrige.rk import Krige

param_dict = {
    "method": ["ordinary", "universal"],
    "variogram_model": ["linear", "power", "gaussian", "spherical"],
    "nlags": [4, 6, 8],
    "weight": [True, False]
}

estimator = RandomizedSearchCV(estimator = Krige(), param_distributions = param_dict, n_iter = 100, cv = 5, verbose=2, n_jobs = -1)


# run the gridsearch
estimator.fit(X=np.array(coords), y=norma)


if hasattr(estimator, "best_score_"):
    print("best_score R² = {:.3f}".format(estimator.best_score_))
    print("best_params = ", estimator.best_params_)

print("\nCV results::")
if hasattr(estimator, "cv_results_"):
    for key in [
        "mean_test_score",
        "param_method",
        "param_variogram_model",
    ]:
        print(" - {} : {}".format(key, estimator.cv_results_[key]))