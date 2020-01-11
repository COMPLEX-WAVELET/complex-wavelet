import dtcwt
import numpy as np

class DtcwtScattering2D:
    """
    The constructor of a dtwct scattering network 
    Attributes:
        J - number of transformations
        L - number of orientations
        m - the depth of the network
    """
    def __init__(self):
        self._transform2D = dtcwt.Transform2d()

    """
    Applies the scattering network transform to an image
    Returns a set of scattering coefficients.
    """

    def transform(self, image, m):
        n = len(image)
        J = int(np.log(n/4)/np.log(2)) + 1
        
        scat_coefficients = []
        image_t = self._transform2D.forward(image, nlevels=J)
        if n >= 4:
            scat_coefficients.append(abs(image_t.lowpass))
        
        if m != 0:
            for j in range(0, J):
                highpass = image_t.highpasses[j]
                for k in range(0, highpass.shape[2]):
                    im = highpass[:,:,k]
                    scat_coefficients = scat_coefficients + self.transform(im, m - 1)
        return scat_coefficients

