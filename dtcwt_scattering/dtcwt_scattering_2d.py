import dtcwt
import numpy as np
from dtcwt.numpy.lowlevel import colfilter, colifilt
from dtcwt.coeffs import biort, qshift
import cmath


class DtcwtScattering2D:
    """
    The constructor of a dtwct scattering transform. 
    """

    def __init__(self):
        self._transform2D = dtcwt.Transform2d()
        self.phases = [[-1]]  # -1 element for S0 as it doesn't have a phase
        self.nlevels = []
        self.scat_layers = []
        self.K = [-1]
        self.scat_ancestor = [
            -1
        ]  # -1 for the zero-th element that doesn't have ancestors

        self.scat_coefficients = []

    def __get_region_magnitude_and_phase(self, region):
        r_max = -1
        phase_max = 0
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                r, phase = cmath.polar(region[i, j])
                if r > r_max:
                    r_max = r
                    phase_max = phase
        return r_max, phase_max

    def __set_region_phase(self, i, j, phase, magnitude, highpass):
        for i1 in range(i, i + 1):
            for j1 in range(j, j + 1):
                highpass[i1, j1] = cmath.rect(magnitude[i1, j1], phase)

    def __restore_phase(self, magnitude, phases):
        highpass = np.zeros((magnitude.shape[0], magnitude.shape[1]), dtype=complex)
        h = magnitude.shape[0]
        k = 0
        for i in range(h):
            for j in range(h):
                highpass[i, j] = cmath.rect(magnitude[i, j], phases[k])
                k += 1

        return highpass

    def __store_phase(self, highpass):
        h = highpass.shape[0]
        phase = []

        for i in range(h):
            for j in range(h):
                _, f = cmath.polar(highpass[i, j])
                phase.append(f)
        self.phases.append(phase)

    """
    Applies the scattering network transform to an image
    Returns a set of scattering coefficients.
    """

    def _transform(self, image, m, enableReconstruction):
        n = len(image)
        J = int(np.log(n / 4) / np.log(2))

        image_t = self._transform2D.forward(image, nlevels=J + 1)
        self.nlevels.append(J + 1)
        self.scat_coefficients.append(image_t.lowpass)
        self.scat_layers.append(m)
        if m != 0:
            ancestor = len(self.scat_coefficients) - 1
            for j in range(0, J):
                highpass = image_t.highpasses[j]
                for k in range(0, highpass.shape[2]):
                    if enableReconstruction:
                        self.__store_phase(highpass[:, :, k])
                        self.scat_ancestor.append(ancestor)
                        self.K.append(k)
                    im = abs(highpass[:, :, k])
                    self._transform(im, m - 1, enableReconstruction)

    def get_scat_layers(self):
        return np.asarray(self.scat_layers)

    """
    Applies log transform to scattering coefficients and returns final result.
    """

    def transform(self, image, m, enableReconstruction=False):
        self.phases = [-1]
        self.nlevels = []
        self.scat_layers = []
        self.scat_ancestor = [-1]
        self.scat_coefficients = []
        self.K = [-1]
        self._transform(image, m, enableReconstruction)
        self.scat_layers = [m - layer for layer in self.scat_layers]
        return self.scat_coefficients

    def __build_pyramid(self, im, levels, is_lowpass, k):
        highpasses = []
        level = 0

        for j in range(levels, 0, -1):
            highpass = np.zeros((2 ** j, 2 ** j, 6), dtype=complex)
            highpasses.append(highpass)
            if 2 ** j == im.shape[0]:
                level = len(highpasses) - 1

        w = self.scat_coefficients[0].shape[0]
        lowpass = np.zeros((w, w), dtype=float)

        if not is_lowpass:
            highpasses[level][:, :, k] = im
        else:
            lowpass = im

        p = dtcwt.Pyramid(lowpass=lowpass, highpasses=tuple(highpasses))
        return p

    def _inverse(self, im, m, is_lowpass):
        k = 0
        levels = 0
        if is_lowpass:
            levels = self.nlevels[m]
        else:
            levels = self.nlevels[self.scat_ancestor[m]]
            k = self.K[m]

        p = self.__build_pyramid(im, levels, is_lowpass, k)
        inversed = self._transform2D.inverse(p)
        if (
            is_lowpass
            and self.phases[m] == -1
            or not is_lowpass
            and self.phases[self.scat_ancestor[m]] == -1
        ):
            return inversed
        if is_lowpass:
            inversed_phase = self.__restore_phase(inversed, self.phases[m])
        else:
            inversed_phase = self.__restore_phase(
                inversed, self.phases[self.scat_ancestor[m]]
            )
        return self._inverse(
            inversed_phase, m if is_lowpass else self.scat_ancestor[m], is_lowpass=False
        )

    def inverse(self, m):
        inversed = self._inverse(self.scat_coefficients[m], m, is_lowpass=True)
        return inversed
