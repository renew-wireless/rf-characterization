#!/usr/bin/python
"""

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 ---------------------------------------------------------------------
"""

import sys
sys.path.append('./IrisUtils/')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from find_lts import *
from ofdmtxrx import *
import time


#########################################
#            Global Parameters          #
#########################################
APPLY_CFO_CORR = 0
APPLY_SFO_CORR = 1
APPLY_PHASE_CORR = 1
frame_count = 0


#########################################
#              Functions                #
#########################################
def wlan_rx(signal, ofdm_params, ofdm_obj):
    global frame_count

    frame_count += 1

    # Init
    n_ofdm_syms = ofdm_params[0]
    cp_len = ofdm_params[1]
    data_cp_len = ofdm_params[2]
    num_sc = ofdm_params[3]
    mod_order = ofdm_params[4]
    fft_offset = ofdm_params[5]
    data_sc = ofdm_params[6]
    pilot_sc = ofdm_params[7]
    pilots_matrix = ofdm_params[8]
    fft_size = ofdm_params[9]
    tx_syms = ofdm_params[10]  # Assume no knowledge of TX other than basic params (e.g., 64 different symbols in 64QAM)

    const = np.zeros((len(data_sc), n_ofdm_syms))
    constp = np.zeros((len(pilot_sc), n_ofdm_syms))
    const.fill(np.nan)
    constp.fill(np.nan)
    H = np.zeros((num_sc))
    x_ax = np.zeros((num_sc))
    phase_error = np.zeros((1, n_ofdm_syms))

    payload_len = n_ofdm_syms * (data_cp_len + num_sc)  # Anritsu 802.11a 54 Mbps signals: 3040 samples (38 symbols)
    # plt.figure(2)
    # plt.plot(np.real(signal))
    # plt.show()

    # Generate an LTS for reference
    lts_sym, lts_freq = generate_training_seq(preamble_type='lts', cp=cp_len, upsample=1)
    lts_syms_len = len(lts_sym)

    # Find LTS peaks (in case LTSs were sent)
    lts_thresh = 0.8
    a, b, peaks0 = find_lts(signal, thresh=lts_thresh, flip=True)
    corr = peaks0

    # Check if LTS found
    if not a:
        #print("SISO_OFDM: No LTS Found!")
        return corr, lts_thresh, const, constp, x_ax, H, phase_error, tx_syms

    # If beginning of frame was not captured in current buffer
    if (a - lts_syms_len) < 0:
        #print("TOO EARLY... CONTINUE! ")
        return corr, lts_thresh, const, constp, x_ax, H, phase_error, tx_syms

    # Decode signal (focus on RF chain A only for now)
    rxSignal = signal
    payload_start = a + 1
    #print("PayloadStart: {}".format(payload_start))

    payload_end = payload_start + payload_len  # Payload_len == (n_ofdm_syms * (num_sc + data_cp_len))
    lts_start = a - lts_syms_len + 1  # where LTS-CP start

    # Apply CFO Correction
    if APPLY_CFO_CORR:
        coarse_cfo_est = ofdm_obj.cfo_correction(rxSignal, lts_start, lts_syms_len, fft_offset)
    else:
        coarse_cfo_est = 0

    correction_vec = np.exp(-1j * 2 * np.pi * coarse_cfo_est * np.array(range(0, len(rxSignal))))
    rxSignal_cfo = rxSignal * correction_vec

    # Channel estimation
    # Get LTS again (after CFO correction)
    lts = rxSignal_cfo[lts_start: lts_start + lts_syms_len]

    # Verify number of samples
    if len(lts) != 160:
        print("INCORRECT START OF PAYLOAD... CONTINUE!")
        return corr, lts_thresh, const, constp, x_ax, H, phase_error, tx_syms

    lts_1 = lts[-64 + -fft_offset + np.array(range(97, 161))]
    lts_2 = lts[-fft_offset + np.array(range(97, 161))]

    # Average 2 LTS symbols to compute channel estimate
    chan_est = np.fft.ifftshift(lts_freq) * (np.fft.fft(lts_1) + np.fft.fft(lts_2))/2

    # Assert sample position
    # NOTE: If packet found is not fully captured in current buffer read, ignore it and continue...
    if len(rxSignal_cfo) >= payload_end:
        # Retrieve payload symbols
        payload_samples = rxSignal_cfo[payload_start: payload_end]
    else:
        print("TOO LATE... CONTINUE! ")
        return corr, lts_thresh, const, constp, x_ax, H, phase_error, tx_syms

    # Assert
    if len(payload_samples) != ((num_sc + data_cp_len) * n_ofdm_syms):
        print("INCORRECT START OF PAYLOAD... CONTINUE!")
        return corr, lts_thresh, const, constp, x_ax, H, phase_error, tx_syms
    else:
        payload_samples_mat_cp = np.reshape(payload_samples, ((num_sc + data_cp_len), n_ofdm_syms), order="F")

    # Remove cyclic prefix
    payload_samples_mat = payload_samples_mat_cp[data_cp_len - fft_offset + 1 + np.array(range(0, num_sc)), :]

    # FFT
    rxSig_freq = np.fft.fft(payload_samples_mat, n=fft_size, axis=0)

    # Equalizer
    chan_est_tmp = chan_est.reshape(len(chan_est), 1, order="F")
    rxSig_freq_eq = rxSig_freq / np.matlib.repmat(chan_est_tmp, 1, n_ofdm_syms)

    # Apply SFO Correction
    if APPLY_SFO_CORR:
        rxSig_freq_eq = ofdm_obj.sfo_correction(rxSig_freq_eq, pilot_sc, pilots_matrix, n_ofdm_syms)
    else:
        sfo_corr = np.zeros((num_sc, n_ofdm_syms))

    # Apply phase correction
    if APPLY_PHASE_CORR:
        phase_error = ofdm_obj.phase_correction(rxSig_freq_eq, pilot_sc, pilots_matrix)
        phase_error_to_plot = phase_error

    else:
        phase_error_to_plot = ofdm_obj.phase_correction(rxSig_freq_eq, pilot_sc, pilots_matrix)
        phase_error = np.zeros((1, n_ofdm_syms))

    phase_corr_tmp = np.matlib.repmat(phase_error, fft_size, 1)
    phase_corr = np.exp(-1j * phase_corr_tmp)
    rxSig_freq_eq_phase = rxSig_freq_eq * phase_corr
    rxSymbols_mat = rxSig_freq_eq_phase[data_sc, :]
    rxPilots_mat = rxSig_freq_eq_phase[pilot_sc, :]

    # PLOTTING SECTION
    rx_H_est_plot = np.squeeze(np.matlib.repmat(complex('nan'), 1, len(chan_est)))
    rx_H_est_plot[data_sc] = np.squeeze(chan_est[data_sc])
    rx_H_est_plot[pilot_sc] = np.squeeze(chan_est[pilot_sc])
    x_ax = (20 / num_sc) * np.array(range(-(num_sc // 2), (num_sc // 2)))  # add 5 on each side for visualization

    # Rename
    const = rxSymbols_mat
    constp = rxPilots_mat
    H = rx_H_est_plot

    return corr, lts_thresh, const, constp, x_ax, H, phase_error_to_plot, tx_syms


def wlan_rx_setup():
    #########################
    #    OFDM Parameters    #
    #########################
    # Seems like these are the parameters for the 802.11a 54Mbps (64QAM) signal from the Anritsu SigGen (IQproducer)
    nOFDMsym = 38
    ltsCpLen = 32
    dataCpLen = 16
    modOrder = 16
    fftOffset = 3
    fftSize = 64
    nSC = 64

    # Data and Pilot Subcarriers
    # data_sc = [1:6, 8:20, 22:26, 38:42, 44:56, 58:63];
    # pilot_sc = [7, 21, 43, 57]
    data_sc = list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)) + \
              list(range(38, 43)) + list(range(44, 57)) + list(range(58, 64))
    pilot_sc = [7, 21, 43, 57]
    n_data_syms = nOFDMsym * len(data_sc)  # One data sym per data-subcarrier per ofdm symbol
    pilots = np.array([1, 1, -1, 1]).reshape(4, 1, order="F")
    pilots_matrix = np.matlib.repmat(pilots, 1, nOFDMsym)

    #########################
    #      OFDM Object      #
    #########################
    ofdm_obj = ofdmTxRx()

    #########################
    #         Other         #
    #########################
    # Tx Sym Location
    val = range(0, modOrder)
    tx_syms = np.zeros(len(val), dtype=complex)
    for x in range(len(val)):
        if modOrder == 2:
            tx_syms[x] = ofdm_obj.bpsk_mod(val[x])
        elif modOrder == 4:
            tx_syms[x] = ofdm_obj.qpsk_mod(val[x])
        elif modOrder == 16:
            tx_syms[x] = ofdm_obj.qam16_mod(val[x])
        elif modOrder == 64:
            tx_syms[x] = ofdm_obj.qam64_mod(val[x])

    ofdm_params = [nOFDMsym, ltsCpLen, dataCpLen, nSC, modOrder, fftOffset, data_sc, pilot_sc, pilots_matrix, fftSize, tx_syms]

    return ofdm_params, ofdm_obj


#########################################
#                 Main                  #
#########################################
def main():

    #########################
    # Load captured samples #
    #########################
    array_real, array_imag = numpy.loadtxt('outfile.txt', unpack=True)
    array = array_real + 1j * array_imag

    #########################
    #        Setup          #
    #########################
    ofdm_params, ofdm_obj = wlan_rx_setup()

    #########################
    #     Main Function     #
    #########################
    wlan_rx(array, ofdm_params, ofdm_obj)


if __name__ == '__main__':
    main()
