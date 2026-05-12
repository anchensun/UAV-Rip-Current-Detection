import numpy as np
import pywt
import scipy.ndimage as ndimage
import scipy.optimize as optimize

def connection_coefficients(wav, order):

    ctol = 1e-15  # Tolerance for coefficients
    etol = 1e-4  # Tolerance for eigenvalues

    lo = wav.rec_lo
    len_lo = len(lo)
    dim = 2 * len_lo - 3
    matrix = np.zeros((dim, dim))
    for m in range(dim):
        for n in range(dim):
            tmp = 0.
            for p, lo_p in enumerate(lo):
                idx = m - 2 * n + p + len_lo - 2
                if (idx >= 0 and idx < len_lo):
                    tmp += lo_p * lo[idx]
            if np.abs(tmp) > ctol:
                matrix[n, m] = tmp
    eval, evec = np.linalg.eig(matrix)
    sigma = 1. / float(2 ** order)
    ev_found = False
    for i, ev in enumerate(eval):
        if (np.abs(np.real(ev) - sigma) < etol) and (np.abs(np.imag(ev) < 1e-14)):
            ev_found = True
            break
 
    coeffs = None
    if ev_found:

        norm_factors = np.array([float(-len_lo + 2 + p) for p in range(dim)]) ** order
        if coeffs == None:
            coeffs = np.sum(norm_factors)
        else:
            coeffs /= np.sum(norm_factors * coeffs)
        tmp = (np.prod(np.arange(1, order + 1))) * np.power(-1., order)
        coeffs *= tmp
    return coeffs


class HighOrderRegularizerConv:

    ORDER_MAX = 6  # Max order.

    def __init__(self, wav):

        self.mode = 'periodization'
        self.wav = wav
        self.wavswap = pywt.Wavelet(
            name='{}_swap'.format(self.wav.name),
            filter_bank=self.wav.inverse_filter_bank,
        )
        self.coeffs = {k: connection_coefficients(wav, k) for k in range(self.ORDER_MAX)}
        self.regularizers = {'l2norm': self._l2norm_gradient,
                             'hornschunck': self._hornschunck_gradient,
                             }

    def evaluate(self, C1, C2, regul_type='l2norm'):
        levels = len(C1) - 1
        U1 = pywt.waverec2(C1, self.wav, mode=self.mode)
        U2 = pywt.waverec2(C2, self.wav, mode=self.mode)
        [resultgrad1, resultgrad2] = self.regularizers[regul_type](U1, U2)
        grad1 = pywt.wavedec2(resultgrad1, self.wavswap, level=levels, mode=self.mode)
        grad2 = pywt.wavedec2(resultgrad2, self.wavswap, level=levels, mode=self.mode)
        result = 0.
        for c, g in zip([C1, C2], [grad1, grad2]):
            result += np.dot(c[0].ravel(), g[0].ravel())
            for cd, gd in zip(c[1:], g[1:]):
                for cdd, gdd in zip(cd, gd):
                    result += np.dot(cdd.ravel(), gdd.ravel())
        return 0.5 * result, (resultgrad1, resultgrad2)

    def _l2norm_gradient(self, U1, U2):

        grad = []
        result = 0.
        c0 = self.coeffs[0]
        result = [self.convolve_separable(U, c0, c0) for U in [U1, U2]]
        return result

    def _hornschunck_gradient(self, U1, U2):

        c0 = self.coeffs[0]
        c2 = self.coeffs[2]
        result = [-(self.convolve_separable(U, c2, c0) + self.convolve_separable(U, c0, c2))
                  for U in [U1, U2]]
        return result

    @staticmethod
    def convolve_separable(x, filter1, filter2, origin1=0, origin2=0):
        tmp = ndimage.filters.convolve(x, filter1.reshape(1, -1), mode='wrap', origin=origin1)
        return ndimage.filters.convolve(tmp, filter2.reshape(-1, 1), mode='wrap', origin=origin2)
