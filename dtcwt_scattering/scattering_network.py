import dtcwt
from numpy import log
from numba import njit
from time import time

transform = dtcwt.Transform2d()
dtcwt.push_backend('numpy')


def scattering_network(scattering_vector,image,m, M):
  n=len(image)
  J=int(log(n/4)/log(2))
  image_t=transform.forward(image, nlevels=J+1)
  if n>4 and m<M:
    for j in range(J):
      for theta in range(image_t.highpasses[j].shape[2]):
        scattering_vector=scattering_network(scattering_vector,abs(image_t.highpasses[j][:,:,theta]),m+1, M)
  return([abs(image_t.lowpass)]+scattering_vector)

def filled(x):
    return scattering_network([], x, 0, 2)
