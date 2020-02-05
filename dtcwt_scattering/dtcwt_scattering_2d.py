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
        self.phases = [-1]  # -1 element for S0 as it doesn't have a phase
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

    def __store_phase(self, highpass):
        h = highpass.shape[0]
        w = highpass.shape[1]
        r = [0] * 4
        phase = [0] * 4
        h = int(h / 2)
        w = int(w / 2)
        r[0], phase[0] = self.__get_region_magnitude_and_phase(highpass[:h, :w])
        r[1], phase[1] = self.__get_region_magnitude_and_phase(highpass[:h, w:])
        r[2], phase[2] = self.__get_region_magnitude_and_phase(highpass[h:, :w])
        r[3], phase[3] = self.__get_region_magnitude_and_phase(highpass[h:, w:])

        self.phases.append(phase)

    def __set_region_phase(self, region, phase, highpass):
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                highpass[i, j] = cmath.rect(region[i, j], phase)

    def __restore_phase(self, magnitude, phases):
        h = magnitude.shape[0]
        w = magnitude.shape[1]
        h = int(h / 2)
        w = int(w / 2)
        highpass = np.ndarray(shape=(h, w), dtype=complex)
        self.__set_region_phase(magnitude[:h, :w], phases[0], highpass)
        self.__set_region_phase(magnitude[:h, w:], phases[1], highpass)
        self.__set_region_phase(magnitude[h:, :w], phases[2], highpass)
        self.__set_region_phase(magnitude[h:, w:], phases[3], highpass)
        return highpass

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

    def __build_pyramid(self, m, im, is_lowpass):
        highpasses = []
        level = 0
        for j in range(self.nlevels[m], 0, -1):
            highpass = np.zeros((2 ** j, 2 ** j, 6), dtype=complex)
            highpasses.append(highpass)
            if 2 ** j == im.shape[0]:
                level = len(highpasses) - 1
        w = self.scat_coefficients[m].shape[0]
        lowpass = np.zeros((w, w), dtype=float)
        if not is_lowpass:
            highpasses[level][:, :, self.K[m]] = im
        else:
            lowpass = im

        p = dtcwt.Pyramid(lowpass=lowpass, highpasses=tuple(highpasses))
        return p

    def _inverse(self, im, m, layer_to_reconstruct, current_layer):
        is_lowpass = True if layer_to_reconstruct == current_layer else False
        p = self.__build_pyramid(m, im, is_lowpass=is_lowpass)
        inversed = self._transform2D.inverse(p)

        if self.phases[m] == -1:  # 0th level, end of reconstruction
            return inversed
        inversed_phase = self.__restore_phase(inversed, self.phases[m])

        return self._inverse(
            inversed_phase,
            self.scat_ancestor[m],
            layer_to_reconstruct,
            layer_to_reconstruct - 1,
        )

    def inverse(self, m):
        layer_to_reconstruct = self.scat_layers[m]
        inversed = self._inverse(
            self.scat_coefficients[m], m, layer_to_reconstruct, layer_to_reconstruct
        )
        return inversed
