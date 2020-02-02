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
        self.J = [-1]
        self.K = [-1]
        self.scat_layers = []
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
        r[0], phase[0] = self.__get_region_magnitude_and_phase(highpass[:h, :w])
        r[1], phase[1] = self.__get_region_magnitude_and_phase(highpass[:h, w:])
        r[2], phase[2] = self.__get_region_magnitude_and_phase(highpass[h:, :w])
        r[3], phase[3] = self.__get_region_magnitude_and_phase(highpass[h:, w:])

        self.phases.append(phase)

    def __set_region_phase(self, region, phase, highpass):
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                highpass[i, j] = cmath.rect(region[i, j], phase)

    def __restore_highpass(self, magnitude, phases):
        h = magnitude.shape[0]
        w = magnitude.shape[1]
        highpass = np.ndarray(shape=(h, w), dtype=complex)
        self.__set_region_phase(magnitude[:h, :w], phases[0], highpass)
        self.__set_region_phase(magnitude[:h, w:], phases[1], highpass)
        self.__set_region_phase(magnitude[h:, :w], phases[2], highpass)
        self.__set_region_phase(magnitude[h:, w:], phases[3], highpass)
        return highpass

    def __c2q(self, highpass):
        x = np.zeros(
            (highpass.shape[0] << 1, highpass.shape[1] << 1), dtype=highpass.real.dtype
        )
        sc = np.sqrt(0.5)
        P = highpass[:, :] * sc
        Q = highpass[:, :] * sc

        x[0::2, 0::2] = P.real  # a = (A+C)*sc
        x[0::2, 1::2] = P.imag  # b = (B+D)*sc
        x[1::2, 0::2] = Q.imag  # c = (B-D)*sc
        x[1::2, 1::2] = -Q.real  # d = (C-A)*sc
        return x

    def __inverse_highpass(self, highpass, idx):
        _, g0o, _, g1o = biort("near_sym_a")
        _, _, g0a, g0b, _, _, g1a, g1b = qshift("qshift_a")

        current_level = self.J[idx] + 1

        k = self.K[idx]
        lh = [0, 5]
        hl = [2, 3]
        hh = [1, 4]

        Z = self.__c2q(highpass)
        while current_level >= 2:
            y1 = None
            y2 = None
            if k in lh:
                y1 = colifilt(Z, g1b, g1a)
            if k in hl:
                y2 = colifilt(Z, g0b, g0a)
            elif k in hh:
                y2 = colifilt(Z, g1b, g1a)

            if y1 is not None:
                Z = colifilt(y1.T, g0b, g0a).T
            elif y2 is not None:
                Z = colifilt(y2.T, g1b, g1a).T
            current_level -= 1

        y1 = None
        y2 = None
        if k in lh:
            y1 = colfilter(Z, g1o)
        elif k in hl:
            y2 = colfilter(Z, g0o)
        elif k in hh:
            y2 = colfilter(Z, g1o)

        if y1 is not None:
            Z = colfilter(y1.T, g0o).T
        elif y2 is not None:
            Z = colfilter(y2.T, g1o).T

        inversed = Z
        return inversed

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
                        self.J.append(j)
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
        self.J = [-1]
        self.K = [-1]
        self.scat_layers = []
        self.scat_ancestor = [-1]
        self.scat_coefficients = []
        self._transform(image, m, enableReconstruction)
        self.scat_layers = [m - layer for layer in self.scat_layers]
        return self.scat_coefficients

    def inverse_lowpass(self, scat_coef, idx):
        _, g0o, _, _ = biort("near_sym_a")
        _, _, g0a, g0b, _, _, _, _ = qshift("qshift_a")

        Z = scat_coef
        current_level = self.nlevels[idx]
        while current_level >= 2:
            y1 = colifilt(Z, g0b, g0a)
            Z = (colifilt(y1.T, g0b, g0a)).T
            current_level -= 1

        y1 = colfilter(Z, g0o)
        Z = (colfilter(y1.T, g0o)).T
        inversed = Z
        return inversed

    def inverse(self, s, m):
        # print("Phase size: ", len(self.phases))
        # print("Layers size:", len(self.scat_layers))

        # print("J size: ", len(self.J))
        # print("K size: ", len(self.K))

        inversed_s_m = self.inverse_lowpass(s[m], m)
        m_1 = self.scat_ancestor[m]

        # print(
        #    "For coefficient from layer "
        #    + str(self.scat_layers[m])
        #    + ", coefficient ancestor is "
        #    + str(self.scat_layers[m_1])
        # )
        if m_1 == -1:
            return inversed_s_m

        inversed_s_m_1 = self.inverse_lowpass(s[m_1], m_1)
        restored_highpass = self.__restore_highpass(inversed_s_m, self.phases[m])
        inversed_highpass = self.__inverse_highpass(restored_highpass, m)

        return inversed_highpass + inversed_s_m_1

