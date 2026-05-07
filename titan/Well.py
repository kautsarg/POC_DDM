import numpy as np
from scipy.signal import lfilter

class Well:
    def __init__(self, time_npr, well_3d_npr, well_3d_temp_npr, well_3d_gain, well_3d_lin, idx_start, idx_settled, idx_end, idx_active):
        self.time_npr = time_npr
        self.well_3d_npr = well_3d_npr
        self.well_3d_temp_npr = well_3d_temp_npr
        self.well_3d_gain = well_3d_gain
        self.well_3d_lin = well_3d_lin
        self.idx_start = idx_start
        self.idx_settled = idx_settled
        self.idx_end = idx_end
        self.idx_active = idx_active

    @property
    def time(self):
        return self.time_npr[self.idx_settled:self.idx_end] - self.time_npr[self.idx_start]

    @property
    def time_min(self):
        return self.time / 60

    @property
    def well_nrows(self):
        return self.well_3d.shape[0]

    @property
    def well_ncols(self):
        return self.well_3d.shape[1]

    @property
    def well_temp_nrows(self):
        return self.well_3d_temp_npr.shape[0]

    @property
    def well_temp_ncols(self):
        return self.well_3d_temp_npr.shape[1]

    @property
    def well_2d_npr(self):
        n_time = self.well_3d_npr.shape[2]
        return self.well_3d_npr.reshape(-1, n_time, order='C').T

    @property
    def well_2d_temp_npr(self):
        return self.well_3d_temp_npr.reshape(-1, self.well_3d_temp_npr.shape[2]).T

    # version 1
    # @property # this is definitely wrong
    # def well_temp_mean(self):  # TODO: adapt: this is linear but all time range
    #     a = 24328.12
    #     b = -0.010848
    #     c = np.min(self.well_3d_temp_npr)-0.1
    #     new = 1 / b * np.log((self.well_3d_temp_npr - c) / a)
    #     temp_2d = new.reshape(-1, new.shape[2]).T
    #     return np.mean(temp_2d, axis=1)
    #
    # @property #and this is the same
    # def well_temp_mean2D(self):  # TODO: adapt: this is linear but all time range
    #     a = 24328.12
    #     b = -0.010848
    #     c = np.min(self.well_3d_temp_npr)-0.1
    #     new = 1 / b * np.log((self.well_3d_temp_npr - c) / a)
    #     return new.reshape(-1, new.shape[2]).T

    # version 2 from the microcontroller
    @property
    def well_temp_lin2d(self):
        temp_2d = self.well_3d_temp_npr.reshape(-1, self.well_3d_temp_npr.shape[2]).T
        a = 1.682e-5
        b = 32.645
        c = 7
        lin = 10 * b * np.log((temp_2d - c) / a)
        lin_plt = - (lin - lin[0])
        return lin_plt

    @property
    def well_temp_mean_then_lin(self):
        temp_2d = self.well_3d_temp_npr.reshape(-1, self.well_3d_temp_npr.shape[2]).T
        temp_mean = np.mean(temp_2d, axis=1)
        a = 1.682e-5
        b = 32.645
        c = 7
        lin = 10 * b * np.log((temp_mean - c) / a)
        lin_plt = - (lin - lin[0])
        return lin_plt

    @property
    def well_temp_2D_NEW(self):
        a = 1913.44
        b = 26.68
        c = np.min(self.well_3d_temp_npr)-0.1
        d = 0.76
        new = -(1 / b) * np.log((self.well_3d_temp_npr - c) / a) + d
        temp_2d = new.reshape(-1, new.shape[2]).T
        return temp_2d

    @property
    def well_temp_mean_NEW(self):
        idx_active_temp = (np.mean(self.well_2d_temp_npr, axis=0) > 10)
        return np.mean(self.well_temp_2D_NEW[:, idx_active_temp], axis=1)

    @property
    def well_3d_nl(self):
        return self.well_3d_npr[:, :, self.idx_settled:self.idx_end]

    @property
    def well_2d_nl(self):
        n_time = self.well_3d_nl.shape[2]
        return self.well_3d_nl.reshape(-1, n_time, order='C').T

    @property
    def well_2d_nl_bs(self):
        return self.well_2d_nl - self.well_2d_nl[0, :]

    @property
    def well_2d_nl_bs_active(self):
        return self.well_2d_nl_bs[:, self.idx_active]

    @property
    def well_2d_nl_bs_active_mean(self):
        return np.mean(self.well_2d_nl_bs_active, axis=1)

    @property
    def well_2d_nl_active(self):
        return self.well_2d_nl[:, self.idx_active]

    @property
    def well_2d_nl_active_mean(self):
        return np.mean(self.well_2d_nl_active, axis=1)

    @property
    def well_3d(self):
        return self.well_3d_lin[:, :, self.idx_settled:self.idx_end]

    @property
    def well_2d(self):
        n_time = self.well_3d.shape[2]
        return self.well_3d.reshape(-1, n_time, order='C').T

    @property
    def well_2d_bs(self):
        return self.well_2d - self.well_2d[0, :]

    @property
    def well_2d_active(self):
        return self.well_2d[:, self.idx_active]

    @property
    def well_2d_bs_active(self):
        return self.well_2d_bs[:, self.idx_active]

    @property
    def well_2d_bs_active_mean(self):
        return np.mean(self.well_2d_bs_active, axis=1)

    def well_2d_bs_filt(self, filt_order=10):
        return lfilter(b=np.ones(filt_order) / filt_order, a=[1], x=self.well_2d_bs.T).T

    def well_2d_bs_active_filt(self, filt_order=10):
        return lfilter(b=np.ones(filt_order) / filt_order, a=[1], x=self.well_2d_bs_active.T).T

    def well_2d_bs_active_mean_filt(self, filt_order=10):
        return lfilter(b=np.ones(filt_order) / filt_order, a=[1], x=self.well_2d_bs_active_mean)

    def add_known_results_dna_TB(self, sample=None, cartridge=None, posneg=None, pcr_ttp=None):
        # sample = 1 to 51
        # cartridge = 1/2
        # posneg = 0/1
        # pcr_ttp int
        self.sample = sample
        self.cartridge = cartridge
        self.posneg = posneg
        self.pcr_ttp = pcr_ttp
        return

