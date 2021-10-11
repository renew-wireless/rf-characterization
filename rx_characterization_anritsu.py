#!/usr/bin/python
"""

 rx_characterization_anritsu.py


    Currently have one 15dB attenuator

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 ---------------------------------------------------------------------
"""

import sys
sys.path.append('./IrisUtils/')
sys.path.append('./IrisUtils/LTE/')

import signal
import time
from functools import partial
import threading
import collections
import SoapySDR
from SoapySDR import *              # SOAPY_SDR_ constants
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from type_conv import *
from print_sensor import *
from AnritsuRemote import AnritsuRemote
from wlan_rx import *
from optparse import OptionParser
from digital_rssi import *
from MyFuncAnimation import *
from bandpower import *
from fft_power import *
from macros import *
from file_rdwr import *
import csv
import scipy.io as sio

plt.style.use('ggplot')

#########################################
#            Global Parameters          #
#########################################
fig = None
line1 = None
line2 = None
line3 = None
line4 = None
line5 = None
line6 = None
line7 = None
line8 = None
line9 = None
line10 = None
line11 = None
line12 = None

numBufferSamps = 1000
rssiPwrBuffer = collections.deque(maxlen=numBufferSamps)
timePwrBuffer = collections.deque(maxlen=numBufferSamps)
freqPwrBuffer = collections.deque(maxlen=numBufferSamps)
noisPwrBuffer = collections.deque(maxlen=numBufferSamps)
rssiPwrBuffer_fpga = collections.deque(maxlen=numBufferSamps)


#########################################
#              Functions                #
#########################################
def init():
    """ Initialize plotting objects """
    global fig, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14

    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    line5.set_data([], [])
    line6.set_data([], [])
    line7.set_data([], [])
    line8.set_data([], [])
    line9.set_data([], [])
    line10.set_data([], [])
    line11.set_data([], [])
    line12.set_data([], [])
    line13.set_data([], [])
    line14.set_data([], [])
    return line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14


def create_plots(FIG_LEN, rate, fft_size):
    global fig, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14
    """
    Initialize plots
    """

    # Select plots
    waveform = True
    amplitude = True
    pwr_fft = True
    pwr_realtime = True
    correlation = True
    const_and_chan = True

    num_plots = waveform + amplitude + pwr_fft + pwr_realtime + correlation + const_and_chan

    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(10, 20), dpi=120)
    fig.subplots_adjust(hspace=.4, top=.97, bottom=.03)
    gs = gridspec.GridSpec(ncols=4, nrows=num_plots)
    subplot_count = 0

    if waveform:
        ax1 = fig.add_subplot(gs[subplot_count, :])
        ax1.grid(True)
        ax1.set_title('Waveform capture')
        title = ax1.text(0.5, 1, '|', ha="center")
        ax1.set_ylabel('Signal')
        ax1.set_xlabel('Sample index')
        line1, = ax1.plot([], [], label='ChA I', animated=True)
        ax1.set_ylim(-1, 1)
        ax1.set_xlim(0, FIG_LEN)
        ax1.legend(fontsize=10)
        subplot_count += 1

    if amplitude:
        ax2 = fig.add_subplot(gs[subplot_count, :])
        ax2.grid(True)
        ax2.set_xlabel('Sample index')
        ax2.set_ylabel('Amplitude')
        line2, = ax2.plot([], [], label='ChA', animated=True)
        ax2.set_ylim(-2, 2)
        ax2.set_xlim(0, FIG_LEN)
        ax2.legend(fontsize=10)
        subplot_count += 1

    if pwr_fft:
        ax3 = fig.add_subplot(gs[subplot_count, :])
        ax3.grid(True)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power (dB)')
        line3, = ax3.plot([], [], label='FFT ChA', animated=True)
        ax3.set_ylim(-110, 0)
        freqScale = np.arange(-rate / 2, rate / 2, rate / fft_size)
        # freqScale = np.arange(-rate / 2 / 1e6, rate / 2 / 1e6, rate / fft_size / 1e6)[:fft_size]
        ax3.set_xlim(freqScale[0], freqScale[-1])
        ax3.legend(fontsize=10)
        subplot_count += 1

    if pwr_realtime:
        ax4 = fig.add_subplot(gs[subplot_count, :])
        ax4.grid(True)
        ax4.set_xlabel('Real-Time Samples')
        ax4.set_ylabel('Power (dB)')
        line4, = ax4.plot([], [], label='Digital RSSI', animated=True)
        line5, = ax4.plot([], [], label='TimeDomain Sig Pwr', animated=True)
        line6, = ax4.plot([], [], label='FreqDomain Sig Pwr', animated=True, linestyle='dashed')
        line7, = ax4.plot([], [], label='Noise Floor', animated=True)
        line8, = ax4.plot([], [], label='RSSI_FPGA_Pwr', animated=True, linestyle='dashed')
        ax4.set_ylim(-100, 10)
        ax4.set_xlim(0, numBufferSamps * 1.5)
        ax4.legend(fontsize=10)
        subplot_count += 1

    if correlation:
        ax5 = fig.add_subplot(gs[subplot_count, :])
        ax5.grid(True)
        ax5.set_title('Correlation Peaks')
        ax5.set_xlabel('Sample index')
        ax5.set_ylabel('')
        line9, = ax5.plot([], [], label='RFA', animated=True)
        line10, = ax5.plot([], [], '--r', label='Thresh', animated=True)  # markers
        ax5.set_ylim(0, 8)
        ax5.set_xlim(0, FIG_LEN)
        ax5.legend(fontsize=10)
        subplot_count += 1

    if const_and_chan:
        ax6 = fig.add_subplot(gs[subplot_count, 0:2])
        ax6.grid(True)
        ax6.set_title('RX Constellation')
        ax6.set_xlabel('')
        ax6.set_ylabel('')
        line11, = ax6.plot([], [], 'bx', label='Data', animated=True)
        line12, = ax6.plot([], [], 'rx', label='Pilots', animated=True)
        line13, = ax6.plot([], [], 'gx', label='TxSyms', animated=True)
        ax6.set_ylim(-1.5, 1.5)
        ax6.set_xlim(-2.8, 2.8)
        ax6.legend(fontsize=10)

        chan = False
        ax7 = fig.add_subplot(gs[subplot_count, 2:4])
        ax7.grid(True)
        if chan:
            ax7.set_title('Magnitude Channel Estimates')
            ax7.set_xlabel('Baseband Freq.')
            ax7.set_ylabel('')
            line14, = ax7.plot([], [], animated=True)
            ax7.set_ylim(-0.1, 5)
            ax7.set_xlim(-10, 10)
        else:
            # Phase Error Estimate
            ax7.set_title('Phase Error')
            ax7.set_xlabel('OFDM Symbol Index')
            line14, = ax7.plot([], [], animated=True)
            ax7.set_ylim(-3.2, 3.2)
            ax7.set_xlim(0, 38)
        ax7.legend(fontsize=10)


def rx_char_app(rate, fft_size, num_ant, freq, freq_str, txgain, rxgain, num_samps, serial, gain_sweep, specan, gain_table,
                specan_remote, wait_trigger, sig_gen_tx_pwr):
    """

    """
    global fig

    if num_ant == 1:
        chan = [0]
    elif num_ant == 2:
        chan = [0, 1]
    else:
        print("Error: Only 1 or 2 channels supported")
        sys.exit()

    # Device information
    sdr = SoapySDR.Device(dict(serial=serial))
    info = sdr.getHardwareInfo()
    print(info)

    # Settings
    for c in chan:
        print("Writing settings for channel {}".format(c))
        sdr.setBandwidth(SOAPY_SDR_TX, c, 2.5*rate)
        sdr.setSampleRate(SOAPY_SDR_TX, c, rate)
        sdr.setFrequency(SOAPY_SDR_TX, c, "RF", freq-.75*rate)
        sdr.setFrequency(SOAPY_SDR_TX, c, "BB", .75*rate)
        sdr.writeSetting(SOAPY_SDR_TX, c, "CALIBRATE", 'SKLK')

        sdr.setBandwidth(SOAPY_SDR_RX, c, 2.5*rate)
        sdr.setSampleRate(SOAPY_SDR_RX, c, rate)
        sdr.setFrequency(SOAPY_SDR_RX, c, "RF", freq-.75*rate)
        sdr.setFrequency(SOAPY_SDR_RX, c, "BB", .75*rate)
        sdr.writeSetting(SOAPY_SDR_RX, c, "CALIBRATE", 'SKLK')

        sdr.setAntenna(SOAPY_SDR_TX, c, "TRX")
        sdr.setGain(SOAPY_SDR_TX, c, txgain)  # Not really needed but, whatever...
        sdr.setGain(SOAPY_SDR_RX, c, rxgain)
        sdr.setDCOffsetMode(SOAPY_SDR_RX, c, True)

    print("Frequency has been set to %f" % sdr.getFrequency(SOAPY_SDR_RX, 0))
    sdr.writeRegister("RFCORE", 120, 0)

    # Setup RX stream
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, chan)

    # RSSI read setup
    setUpDigitalRssiMode(sdr)

    # Initialize registers to be used by AGC and packet detect FPGA cores
    register_setup(sdr)

    # Stop/Close/Cleanup
    signal.signal(signal.SIGINT, partial(signal_handler, sdr, rxStream, specan, specan_remote))

    # WLAN Rx Setup
    ofdm_params, ofdm_obj = wlan_rx_setup()

    # Animate Function
    if gain_sweep:
        gain_sweep_thread = threading.Thread(target=gain_sweep_fncn, args=(sdr, rate, chan, freq_str, fft_size, rxStream, num_samps, wait_trigger, gain_table, sig_gen_tx_pwr, specan, specan_remote, ofdm_params, ofdm_obj))
        gain_sweep_thread.start()

    else:
        create_plots(num_samps, rate, fft_size)
        if specan_remote:
            specan.siggen_output_onoff('ON')
        anim = MyFuncAnimation(fig, animate, init_func=init, fargs=(sdr, rate, fft_size, rxStream, num_samps, wait_trigger, ofdm_params, ofdm_obj),
                               frames=100, interval=100, blit=True)
        plt.show()

    print("Ctrl+C to stop")
    signal.pause()


def gain_sweep_fncn(sdr, rate, chan, freq_str, fft_size, rxStream, num_samps, wait_trigger, gain_table, sig_gen_tx_pwr, specan, specan_remote, ofdm_params, ofdm_obj):
    """
    Run gain sweep thread
    """

    for idx, tx_pwr in enumerate(sig_gen_tx_pwr):
        filename = "rx_vs_rssi_" + freq_str + "_txPwr" + str(tx_pwr) + "dBm_extAttn15dB.csv"
        first_iter = True
        not_good = True

        if specan_remote:
            while not_good:
                try:
                    if tx_pwr == -999:
                        # Measure Noise Floor
                        specan.siggen_output_onoff('OFF')
                    else:
                        # New output power
                        specan.ampl = tx_pwr
                        specan.siggen_set_output_pwr()
                        specan.siggen_output_onoff('ON')
                    not_good = False

                except:
                    not_good = True
                    continue

        for rx_gain_idx in range(len(gain_table)):
            print("Running TX PWR: {}, FREQ: {}, RX GAIN IDX: {} / {}".format(tx_pwr, freq_str, rx_gain_idx, len(gain_table)))
            for ch in chan:
                # Format: gain_table = [count, total, LNA1, LNA2, ATTN, LNA, TIA, PGA]
                sdr.setGain(SOAPY_SDR_RX, ch, 'LNA1', gain_table[rx_gain_idx][2])
                sdr.setGain(SOAPY_SDR_RX, ch, 'LNA2', gain_table[rx_gain_idx][3])
                sdr.setGain(SOAPY_SDR_RX, ch, 'ATTN', gain_table[rx_gain_idx][4])
                sdr.setGain(SOAPY_SDR_RX, ch, 'LNA', gain_table[rx_gain_idx][5])
                sdr.setGain(SOAPY_SDR_RX, ch, 'TIA', gain_table[rx_gain_idx][6])
                sdr.setGain(SOAPY_SDR_RX, ch, 'PGA', gain_table[rx_gain_idx][7])

                # FIXME - some strange bug... once we change gain, if we don't run setup again RSSI returns crap
                setUpDigitalRssiMode(sdr)

            time.sleep(0.5)

            num_iter = 3
            rssi_list = np.array([0] * num_iter)
            rssi_list2 = np.array([0] * num_iter)
            pwr_list = np.array([0] * num_iter)
            pwr_list2 = np.array([0] * num_iter)
            evm_list = np.array([0] * num_iter)
            evm_list2 = np.array([0] * num_iter)

            iter = 0
            while rssi_list[-1] == 0:
                tmp0, tmp1, rssi_res, evm_res = rx_samples(sdr, rate, fft_size, rxStream, num_samps, wait_trigger, ofdm_params, ofdm_obj)

                if rssi_res[0] == 0:
                    continue

                rssi_list[iter] = rssi_res[0]
                rssi_list2[iter] = rssi_res[1]
                pwr_list[iter] = rssi_res[2]
                pwr_list2[iter] = rssi_res[3]
                evm_list[iter] = evm_res[0]
                evm_list2[iter] = evm_res[1]

            iter += 1

            avg_rssi = np.mean(rssi_list)
            avg_rssi_fpga = np.mean(rssi_list2)
            avg_time_pwr = np.mean(pwr_list)
            avg_freq_pwr = np.mean(pwr_list2)
            avg_evm_pct = np.mean(evm_list)
            avg_evm_snr = np.mean(evm_list2)

            results_final = [gain_table[rx_gain_idx][0],  # Index
                             gain_table[rx_gain_idx][1],  # total gain
                             gain_table[rx_gain_idx][5],  # LNA
                             gain_table[rx_gain_idx][6],  # TIA
                             gain_table[rx_gain_idx][7],  # PGA
                             gain_table[rx_gain_idx][4],  # ATTN
                             gain_table[rx_gain_idx][2],  # LNA1
                             gain_table[rx_gain_idx][3],  # LNA2
                             avg_rssi,                    # RSSI
                             avg_rssi_fpga,               # RSSI_FPGA
                             avg_time_pwr,                # PWR TIME DOMAIN
                             avg_freq_pwr,                # PWR FREQ DOMAIN
                             avg_evm_pct,                 # AVG EVM (%)
                             avg_evm_snr]                 # AVG SNR FROM EVM

            # CSV file
            with open(filename, 'a') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if first_iter:
                    # Write headers
                    writer.writerow(['Index', 'TotalGain', 'LNA', 'TIA', 'PGA', 'ATTN', 'LNA1', 'LNA2', 'RSSI', 'RSSI_FPGA', 'PWR_TIME(LIN)', 'PWR_FREQ(LIN)', 'EVM_PCT', 'EVM_SNR'])
                    writer.writerow(results_final)
                    first_iter = False
                else:
                    writer.writerow(results_final)

    print("Stopped gain_sweep thread!")
    sys.exit()


def animate(i, sdr, rate, fft_size, rxStream, num_samps, wait_trigger, ofdm_params, ofdm_obj):
    global fig, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14
    store_samps = False

    sampsRx, pwr_res, _, ofdm_res, evm_res = rx_samples(sdr, rate, fft_size, rxStream, num_samps, wait_trigger, ofdm_params, ofdm_obj)

    # Results
    f1 = pwr_res[0]
    powerBins = pwr_res[1]
    PWRdBFS = pwr_res[2]
    sigPwr_dB = pwr_res[3]
    fftPower_dB = pwr_res[4]
    noiseFloor = pwr_res[5]
    PWRdBm_fpga = pwr_res[6]
    corr = ofdm_res[0]
    lts_thresh = ofdm_res[1]
    const = ofdm_res[2]
    constp = ofdm_res[3]
    x_ax = ofdm_res[4]
    H = ofdm_res[5]
    phase_err = ofdm_res[6]
    tx_syms = ofdm_res[7]

    # Circular buffer - continuously display data
    rssiPwrBuffer.append(PWRdBFS)
    timePwrBuffer.append(sigPwr_dB)
    freqPwrBuffer.append(fftPower_dB)
    noisPwrBuffer.append(noiseFloor)
    rssiPwrBuffer_fpga.append(PWRdBm_fpga)

    # Fill out data structures with measured data
    line1.set_data(range(sampsRx[0].size), np.real(sampsRx[0]))             # 1: Real of Signal
    line2.set_data(range(sampsRx[0].size), np.abs(sampsRx[0]))              # 2: Amplitude of Signal
    line3.set_data(f1, powerBins)                                           # 3: FFT Power
    line4.set_data(range(len(rssiPwrBuffer)), rssiPwrBuffer)                # 4: Digital RSSI
    line5.set_data(range(len(timePwrBuffer)), timePwrBuffer)                # 5: TimeDomain Sig Pwr
    line6.set_data(range(len(freqPwrBuffer)), freqPwrBuffer)                # 6: FreqDomain Sig Pwr
    line7.set_data(range(len(noisPwrBuffer)), noisPwrBuffer)                # 7: Noise Floor
    line8.set_data(range(len(rssiPwrBuffer_fpga)), rssiPwrBuffer_fpga)      # 8: RSSI_FPGA_Pwr
    line9.set_data(range(len(corr)), np.abs(corr))                          # 9: Correlation Peaks
    line10.set_data(np.linspace(0.0, num_samps, num=1000),
                    (lts_thresh * np.max(corr)) * np.ones(1000))            # 10: Correlation Thresh
    line11.set_data(np.real(const[0:5]), np.imag(const[0:5]))                         # 11: Constellation - Data
    line12.set_data(np.real(constp), np.imag(constp))                       # 12: Constellation - Pilots
    line13.set_data(np.real(tx_syms), np.imag(tx_syms))                     # 13: Constellation - Pilots
    try:
        line14.set_data(range(len(phase_err)), phase_err)
        # line13.set_data(x_ax, np.fft.fftshift(abs(H)))                     # 14: Channel
    except:
        print(H)
        stop = 1
    # Write samples to file
    if store_samps:
        numpy.savetxt('outfile.txt', numpy.column_stack([sampsRx[0].real, sampsRx[0].imag]))

    return line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14


def rx_samples(sdr, rate, fft_size, rxStream, num_samps, wait_trigger, ofdm_params, ofdm_obj):

    debug = False

    # Read samples into this buffer
    sampsRx = [np.zeros(num_samps, np.complex64), np.zeros(num_samps, np.complex64)]
    buff0 = sampsRx[0]  # RF Chain 1
    buff1 = sampsRx[1]  # RF Chain 2

    flags = SOAPY_SDR_END_BURST
    if wait_trigger:
        flags |= SOAPY_SDR_WAIT_TRIGGER
    sdr.activateStream(rxStream,
                       flags,       # flags
                       0,           # timeNs (dont care unless using SOAPY_SDR_HAS_TIME)
                       buff0.size)  # numElems - this is the burst size
    sr = sdr.readStream(rxStream, [buff0, buff1], buff0.size)
    if sr.ret != buff0.size:
        print("Read RX burst of %d, requested %d" % (sr.ret, buff0.size))

    # DC removal
    for i in [0, 1]:
        sampsRx[i] -= np.mean(sampsRx[i])

    #########################
    # Load captured samples #
    #########################
    #print("LOADING DATA")
    #array_real, array_imag = numpy.loadtxt('outfile.txt', unpack=True)
    #array = array_real + 1j * array_imag
    # sio.savemat('outfile.mat', {'array': array})

    # sampsRx_dec = sampsRx[0][1::2]
    # Compute EVM - No knowledge of TX symbols, find closest
    corr, lts_thresh, const, constp, x_ax, H, phase_err, tx_syms = wlan_rx(sampsRx[0], ofdm_params, ofdm_obj)
    evm = np.zeros(const.size)
    const_all = np.reshape(const, const.size, order="F")
    for idx, rx_sym in enumerate(const_all):
        min_idx = np.argmin(np.abs(rx_sym - tx_syms))
        evm[idx] = np.abs(tx_syms[min_idx] - rx_sym)

    avg_evm = np.mean(evm)
    avg_evm_pct = np.mean(100 * evm)
    snr_evm_total = 10*np.log10(1/avg_evm)
    print("EVM: AVG: {}, PCT: {}, SNR: {}".format(avg_evm, avg_evm_pct, snr_evm_total))

    # Magnitude of IQ Samples (RX RF chain A)
    I = np.real(sampsRx[0])
    Q = np.imag(sampsRx[0])
    IQmag = np.mean(np.sqrt(I**2 + Q**2))

    # Retrieve RSSI measured from digital samples at the LMS7, and convert to PWR in dBm
    agc_avg = 3
    rssi, PWRdBFS = getDigitalRSSI(sdr, agc_avg)  # dBFS

    # Compute Power of Time Domain Signal
    sigRms = np.sqrt(np.mean(sampsRx[0] * np.conj(sampsRx[0])))
    sigPwr = np.real(sigRms) ** 2
    sigPwr_dB = 10 * np.log10(sigPwr)
    sigPwr_dBm = 10 * np.log10(sigPwr / 1e-3)

    # Compute Power of Frequency Domain Signal (FFT)
    f1, powerBins, noiseFloor, pks = fft_power(sampsRx[0], rate, num_bins=fft_size, peak=1.0,
                                               scaling='spectrum', peak_thresh=20)
    fftPower = bandpower(sampsRx[0], rate, 0, rate / 2)
    if fftPower <= 0:
        fftPower = 1e-15     # Remove warning
    fftPower_dB = 10 * np.log10(fftPower)
    fftPower_dBm = 10 * np.log10(fftPower / 1e-3)

    # Retrieve RSSI computed in the FPGA
    rssi_fpga = int(sdr.readRegister("IRIS30", FPGA_IRIS030_RD_MEASURED_RSSI))
    Vrms_fpga = (rssi_fpga / 2.0 ** 16) * (1 / np.sqrt(2.0))  # Vrms = Vpeak/sqrt(2) (In Volts) - 16 bit value
    PWRrms_fpga = (Vrms_fpga ** 2.0) / 50.0                   # 50 Ohms load (PWRrms in Watts)
    PWRdBm_fpga = 10.0 * np.log10(PWRrms_fpga) + 30           # P(dBm)=10*log10(Prms/1mW)  OR  P(dBm)=10*log10(Prms)+30

    # print("RSSI: {} \t FPGA: {} \t Time: {} \t Freq: {}".format(rssi, rssi_fpga, sigPwr, fftPower))

    if debug:
        lna_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'LNA')    # ChanA (0)
        tia_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'TIA')    # ChanA (0)
        pga_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'PGA')    # ChanA (0)
        lna1_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'LNA1')    # ChanA (0)
        lna2_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'LNA2')    # ChanA (0)
        attn_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'ATTN')    # ChanA (0)
        print("ATTN: {} \t LNA1: {} \t LNA2: {} \t LNA: {} \t TIA: {} \t PGA: {} ".format(
            attn_rd, lna1_rd, lna2_rd, lna_rd, tia_rd, pga_rd))

    rssi_res = [rssi, rssi_fpga, sigPwr, fftPower]
    pwr_res = [f1, powerBins, PWRdBFS, sigPwr_dB, fftPower_dB, noiseFloor, PWRdBm_fpga]
    ofdm_res = [corr, lts_thresh, const, constp, x_ax, H, phase_err, tx_syms]
    evm_res = [avg_evm_pct, snr_evm_total]

    return sampsRx, pwr_res, rssi_res, ofdm_res, evm_res


def register_setup(sdr):
    # AGC setup
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_ENABLE_FLAG, 0)         # Enable AGC Flag (set to 0 initially)
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_RESET_FLAG, 1)          # Reset AGC Flag
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_IQ_THRESH, 10300)           # Saturation Threshold: 10300 about -6dBm
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_NUM_SAMPS_SAT, 3)           # Number of samples needed to claim sat.
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_MAX_NUM_SAMPS_AGC, 20)      # Threshold at which AGC stops
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_WAIT_COUNT_THRESH, 160)     # Gain settle takes about 20 samps(value=20)
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_RESET_FLAG, 0)          # Clear AGC reset flag
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_BIG_JUMP, 15)           # Drop gain at initial saturation detection
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_SMALL_JUMP, 5)          # Drop gain at subsequent sat. detections
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_RSSI_TARGET, 20)            # RSSI Target for AGC: ideally around 14
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_TEST_GAIN_SETTINGS, 0)  # Enable only for testing gain settings

    # Pkt Detect Setup
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_THRESH, 0)          # RSSI value at which Pkt is detected
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NUM_SAMPS, 5)       # Number of samples needed to detect frame
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_ENABLE, 0)          # Enable packet detection flag
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NEW_FRAME, 0)       # Finished last frame? (set to 0 initially)


def signal_handler(sdr, rxStream, specan, specan_remote, signal, frame):
    global running
    running = False
    print("Exiting Program. Cleanup Streams")
    if rxStream is not None:
        sdr.deactivateStream(rxStream)
        sdr.closeStream(rxStream)
    if specan_remote:
        specan.siggen_output_onoff('OFF')
    sys.exit(0)


#########################################
#                 Main                  #
#########################################
def main():

    parser = OptionParser()
    parser.add_option("--rate", type="float", dest="rate", help="Tx sample rate", default=20e6)  # for 5MHz rate= 7.68e6
    parser.add_option("--fft_size", type="float", dest="fft_size", help="FFT Size", default=512) #2**12)
    parser.add_option("--bw", type="float", dest="bw", help="Bandwidth", default=20)  # 5MHz
    parser.add_option("--num_ant", type="string", dest="num_ant", help="Optional Tx antenna", default=1)
    parser.add_option("--freq", type="string", dest="freq", help="Tx RF freq band (HI=3.6GHz, LO=2.5GHz)", default="LO")
    parser.add_option("--txgain", type="float", dest="txgain", help="Tx Gain (dB) - Iris", default=10.0)
    parser.add_option("--rxgain", type="float", dest="rxgain", help="Init Rx Gain (dB)", default=55.0)
    parser.add_option("--num_samps", type="float", dest="num_samps", help="Number of Samples", default=16384)
    parser.add_option("--serial", type="string", dest="serial", help="serial number of the device", default="RF3E000375")
    parser.add_option("--sig_type", type="string", dest="sig_type", help="Signal Type WLAN/TM3_2", default="WLAN")
    parser.add_option("--gain_sweep", action="store_true", dest="gain_sweep", help="Gain Sweep", default=False)
    parser.add_option("--specan_remote", action="store_true", dest="specan_remote", help="Connect to remote spectrum analyzer for characterization", default=True)
    parser.add_option("--wait-trigger", action="store_true", dest="wait_trigger", help="Wait for a trigger to start a frame",default=False)
    (options, args) = parser.parse_args()

    # Display parameters
    print("\n")
    print("========== RX CHAR PARAMETERS =========")
    print("Transmitting {} signal from board {}".format(options.sig_type, options.serial))
    print("Sample Rate (sps): {}".format(options.rate))
    print("# of Antennas: {}".format(options.num_ant))
    print("Frequency (Hz): {}".format(options.freq))
    print("Gain Sweep? {}".format(options.gain_sweep))
    print("Spectrum Analyzer? {}".format(options.specan_remote))
    print("=======================================")
    print("\n")

    # Create gain table for gain sweep
    if options.freq == "LO":
        freq = 2.5e9
        freq_str = 'LO'
        LNA = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27, 28, 29, 30]
        TIA = [0, 9, 12]
        PGA = list(range(-12, 19+1))
        LNA1 = [33]         # Always ON?
        LNA2 = [0, 17]
        ATTN = [-18, -12, -6]  # Cap at -6
        sig_gen_tx_pwr = [-999, -80, -70, -60, -50, -40]  # -999 == Noise Floor Measurements

    elif options.freq == "HI":
        freq = 3.6e9
        freq_str = 'HI'
        LNA = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27, 28, 29, 30]
        TIA = [0, 9, 12]
        PGA = list(range(-12, 19+1))
        LNA1 = [33]         # Always ON?
        LNA2 = [0, 14]
        ATTN = [-18, -12, -6]  # Cap at -6
        sig_gen_tx_pwr = [-999, -60, -50, -40, -30, -20]

    else:
        print("")
        sys.exit(0)

    count = 0
    gain_table = []
    for idx0 in LNA1:
        for idx1 in LNA2:
            for idx2 in ATTN:
                for idx3 in LNA:
                    for idx4 in TIA:
                        for idx5 in PGA:
                            total = sum([idx0, idx1, idx2, idx3, idx4, idx5])
                            gain_table.append([count, total, idx0, idx1, idx2, idx3, idx4, idx5])
                            count += 1

    print("RX TABLE LENGTH: {}".format(len(gain_table)))

    if options.specan_remote:
        # Initialize Anritsu Remote controller
        specan = AnritsuRemote(freq, options.bw, options.sig_type)
        # Initialize SigGen
        specan.ampl = -40
        specan.app_siggen_setup()
    else:
        specan = []

    rx_char_app(
        rate=options.rate,
        fft_size=options.fft_size,
        num_ant=options.num_ant,
        freq=freq,
        freq_str=freq_str,
        txgain=options.txgain,
        rxgain=options.rxgain,
        num_samps=options.num_samps,
        serial=options.serial,
        gain_sweep=options.gain_sweep,
        specan=specan,
        gain_table=gain_table,
        specan_remote=options.specan_remote,
        wait_trigger=options.wait_trigger,
        sig_gen_tx_pwr=sig_gen_tx_pwr,
    )


if __name__ == '__main__':
    main()