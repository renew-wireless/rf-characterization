#!/usr/bin/python
"""

 tx_characterization_anritsu.py


    Currently have one 15dB attenuator
    Max TX gains are set such that the max value + PAPR of 10dB gives something between 20 and 25dBm TX power

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 ---------------------------------------------------------------------
"""

import sys
sys.path.append('./IrisUtils/')
sys.path.append('./IrisUtils/LTE/')

import time
import signal
from functools import partial
import threading
import SoapySDR
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from SoapySDR import *              # SOAPY_SDR_ constants
from type_conv import *
from print_sensor import *
from AnritsuRemote import AnritsuRemote
from optparse import OptionParser
from scipy.io import loadmat
import LTE5_re
import LTE5_im
import csv
import copy


#########################################
#            Global Parameters          #
#########################################
running = True


#########################################
#              Functions                #
#########################################
def tx_char_app(rate, num_ant, freq, freq_str, gain, serial, sig_type, gain_sweep, specan, gain_table, specan_remote, gain_table_en):
    """

    """
    global running

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
        sdr.setGain(SOAPY_SDR_TX, c, gain)


    # SYNC_DELAYS
    sdr.writeSetting("SYNC_DELAYS", "")

    # Setup TX/RX stream
    txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, chan)  # SOAPY_SDR_CF32
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, chan)

    # Clear HW time for easy debugging
    sdr.setHardwareTime(0)

    # Activate streams
    sdr.activateStream(rxStream)
    sdr.activateStream(txStream)
    '''
    txStream = None
    rxStream = None
    '''
    # Stop/Close/Cleanup
    signal.signal(signal.SIGINT, partial(signal_handler, sdr, txStream, rxStream))
    tx_thread = threading.Thread(target=tx_replay, args=(sdr, sig_type, txStream, rxStream, rate))
    tx_thread.start()

    specan_meas_thread = threading.Thread(target=specan_meas, args=(sdr, gain_sweep, specan, chan, gain_table,
                                                                    specan_remote, freq_str, gain_table_en))
    specan_meas_thread.start()


    print("Ctrl+C to stop")
    signal.pause()

def tx_replayNEW(sdr, sig_type, txStream, rxStream, rate):
    """
    Continuously print sensor information
    """
    # Retrieve signal to TX
    txSignal = generate_signal(sig_type)
    num_samps = len(txSignal)
    print("OBCH Loops will operate on chunks of {} samples".format(num_samps))
    txSignal_ui32 = cfloat2uint32(txSignal, order='QI')

    replay_addr = 0
    sdr.writeRegisters("TX_RAM_A", replay_addr, txSignal_ui32.tolist())
    sdr.writeSetting("TX_REPLAY", str(num_samps))  # this starts transmission

def tx_replay(sdr, sig_type, txStream, rxStream, rate):
    """
    Continuously print sensor information
    """
    global running

    # Retrieve signal to TX
    txSignal = generate_signal(sig_type)
    NUM_SAMPS = len(txSignal)
    print("Loops will operate on chunks of {} samples".format(NUM_SAMPS))

    # 4 milliseconds (in units of nanoseconds)
    txTimeDelta = 10e6  # 4e6  # 19360000
    print("Tx time delta (ticks): {} ".format(SoapySDR.timeNsToTicks(int(txTimeDelta), rate)))

    sdr.activateStream(rxStream, 0, 0)
    sdr.activateStream(txStream)

    waveRxA = np.array([0] * NUM_SAMPS, np.uint32)
    waveRxB = np.array([0] * NUM_SAMPS, np.uint32)
    waveTxA = txSignal
    waveTxB = txSignal

    totalRxSamples = 0
    totalTxSamples = 0
    numIterations = 0
    firstTime = True

    while running:
        ####################
        # RECEIVER LOOP
        ####################
        txTimeNs = 0
        timeNs = 0
        flags = 0
        sr = sdr.readStream(rxStream, [waveRxA, waveRxB], NUM_SAMPS, flags, timeNs)

        if sr.ret < 0:
            # print("Unexpected readStream error {}".format(SoapySDR.errToStr(sr)))
            sys.exit(0)
        elif sr.ret != NUM_SAMPS:
            #print("Unexpected readStream return r != NUM_SAMPS. Returned {}".format(sr.ret))
            sys.exit(0)
        else:
            # print("READ {} SAMPS".format(sr.ret))
            txTimeNs = sr.timeNs + int(txTimeDelta)
            totalRxSamples += sr.ret
            if firstTime:
                print("First receive time {}".format(sr.timeNs))
                print("First transmit time {}".format(txTimeNs))
                firstTime = False

        ####################
        # TRANSMIT LOOP
        ####################
        flags = SOAPY_SDR_HAS_TIME
        if not running:
            flags |= SOAPY_SDR_END_BURST  # End burst on last iter

        st = sdr.writeStream(txStream, [waveTxA, waveTxB], NUM_SAMPS)  # , flags, txTimeNs)
        if st.ret < 0:
            #print("Unexpected writeStream error {}".format(SoapySDR.errToStr(st)))
            sys.exit(0)
        elif st.ret != NUM_SAMPS:
            #print("Unexpected readStream return r != NUM_SAMPS. Returned {}".format(st.ret))
            sys.exit(0)
        else:
            flags = 0
            totalTxSamples += st.ret
            # print("TX   {} SAMPS".format(sr.ret))

        '''
        IAMP = sdr.getGain(SOAPY_SDR_TX, 0, "IAMP")
        PAD = sdr.getGain(SOAPY_SDR_TX, 0, "PAD")
        PA1 = sdr.getGain(SOAPY_SDR_TX, 0, "PA1")
        PA2 = sdr.getGain(SOAPY_SDR_TX, 0, "PA2")
        PA3 = sdr.getGain(SOAPY_SDR_TX, 0, "PA3")
        ATTN = sdr.getGain(SOAPY_SDR_TX, 0, "ATTN")
        print("TX GAINS: PA1 {}, PA2 {}, PA3 {}, ATTN {}, PAD {}, IAMP {}".format(PA1, PA2, PA3, ATTN, PAD, IAMP))
        '''

        numIterations += 1
        if (numIterations % 100) == 0:
            print("Number of Iterations: {}".format(numIterations))

    print("(2) Stopped tx_replay thread!")
    sys.exit()


def specan_meas(sdr, gain_sweep, specan, chan, gain_table, specan_remote, freq_str, gain_table_en):

    global running
    gain_count = 0
    num_reads = 5
    first_iter = True
    #filename = "tx_vs_evm_" + freq_str + "_15dB_ExtAtten_PARTIAL.csv"
    #filename = "tx_vs_evm_" + freq_str + "_newTableDisablePA2_Oct10_2021.csv"
    #filename = "tx_vs_evm_" + freq_str + "_FinalGainTable_Oct10_2021.csv"
    filename = "test.csv"
    res0 = np.empty(num_reads)
    res1 = np.empty(num_reads)
    res2 = np.empty(num_reads)
    res3 = np.empty(num_reads)
    res4 = np.empty(num_reads)
    res5 = np.empty(num_reads)
    res6 = np.empty(num_reads)
    res7 = np.empty(num_reads)

    # Result Indexes
    FREQ_ERR_AVG_HZ = 0
    FREQ_ERR_MAX_HZ = 1
    PWR_OUT_AVG_DBM = 2
    PWR_OUT_MAX_DBM = 3
    PWR_MEAN_AVG_DBM = 4
    PWR_MEAN_MAX_DBM = 5
    EVM_RMS_AVG_PCT = 6
    EVM_RMS_MAX_PCT = 7

    while running:
        ####################
        # SET GAINS
        ####################
        if gain_sweep:
            # Set gains
            # gain_count = 68  # TEST   MAX INDEX: 4511 @ 3.6G, 2651 @ 2.5G
            if gain_count % 10 == 0:
                print("Running: {} / {}".format(gain_count, len(gain_table)))
            if gain_count == len(gain_table):
                # Stop, done at gain_count == len(gain_table)-1
                running = False
                break

            for c in chan:
                sdr.setGain(SOAPY_SDR_TX, c, 'PA1', gain_table[gain_count][2])
                sdr.setGain(SOAPY_SDR_TX, c, 'PA2', gain_table[gain_count][3])
                sdr.setGain(SOAPY_SDR_TX, c, 'PA3', gain_table[gain_count][4])
                sdr.setGain(SOAPY_SDR_TX, c, 'ATTN', gain_table[gain_count][5])
                sdr.setGain(SOAPY_SDR_TX, c, 'IAMP', gain_table[gain_count][6])
                sdr.setGain(SOAPY_SDR_TX, c, 'PAD', gain_table[gain_count][7])
            print("SET GAINS: {}".format(gain_table[gain_count]))
            gain_count += 1
            time.sleep(1)  # Wait for gains to settle

        elif gain_table_en:
            # FORMAT: gain_table.append([total, PA1, PA2, PA3, ATTN, PAD, IAMP])
            if gain_count == len(gain_table):
                # Stop, done at gain_count == len(gain_table)-1
                running = False
                break

            for c in chan:
                sdr.setGain(SOAPY_SDR_TX, c, 'PA1', gain_table[gain_count][1])
                sdr.setGain(SOAPY_SDR_TX, c, 'PA2', gain_table[gain_count][2])
                sdr.setGain(SOAPY_SDR_TX, c, 'PA3', gain_table[gain_count][3])
                sdr.setGain(SOAPY_SDR_TX, c, 'ATTN', gain_table[gain_count][4])
                sdr.setGain(SOAPY_SDR_TX, c, 'IAMP', gain_table[gain_count][6])
                sdr.setGain(SOAPY_SDR_TX, c, 'PAD', gain_table[gain_count][5])
            print("SET GAINS: {}".format(gain_table[gain_count]))
            gain_count += 1
            time.sleep(1)  # Wait for gains to settle

            IAMP = sdr.getGain(SOAPY_SDR_TX, 0, "IAMP")
            PAD = sdr.getGain(SOAPY_SDR_TX, 0, "PAD")
            PA1 = sdr.getGain(SOAPY_SDR_TX, 0, "PA1")
            PA2 = sdr.getGain(SOAPY_SDR_TX, 0, "PA2")
            PA3 = sdr.getGain(SOAPY_SDR_TX, 0, "PA3")
            ATTN = sdr.getGain(SOAPY_SDR_TX, 0, "ATTN")
            print("TX GAINS: PA1 {}, PA2 {}, PA3 {}, ATTN {}, PAD {}, IAMP {}".format(PA1, PA2, PA3, ATTN, PAD, IAMP))

        else:
            # Test with fixed gain settings
            print("*** USING FIXED GAINS ***")
            for c in chan:
                sdr.setGain(SOAPY_SDR_TX, c, 'PA1', 13.7)
                sdr.setGain(SOAPY_SDR_TX, c, 'PA2', 0)
                sdr.setGain(SOAPY_SDR_TX, c, 'PA3', 31)
                sdr.setGain(SOAPY_SDR_TX, c, 'ATTN', -6)
                sdr.setGain(SOAPY_SDR_TX, c, 'IAMP', 0)
                sdr.setGain(SOAPY_SDR_TX, c, 'PAD', 40)
            print("SET GAINS: {}".format(gain_table[gain_count]))
            time.sleep(1)  # Wait for gains to settle

        ####################
        # MEASUREMENT
        ####################
        if specan_remote:
            iter_idx = 0
            while iter_idx < num_reads:
                # print(" Modulation Measurements ")
                try:
                    specan.modulation_measurements()
                except:
                    print("FAILED, RETRY !!!")
                    continue

                # Record all results
                evm_vec = specan.evm_vec
                evm_vec = evm_vec.split(',')
                res0[iter_idx] = float(evm_vec[FREQ_ERR_AVG_HZ])
                res1[iter_idx] = float(evm_vec[FREQ_ERR_MAX_HZ])
                res2[iter_idx] = float(evm_vec[PWR_OUT_AVG_DBM])
                res3[iter_idx] = float(evm_vec[PWR_OUT_MAX_DBM])
                res4[iter_idx] = float(evm_vec[PWR_MEAN_AVG_DBM])
                res5[iter_idx] = float(evm_vec[PWR_MEAN_MAX_DBM])
                res6[iter_idx] = float(evm_vec[EVM_RMS_AVG_PCT])
                res7[iter_idx] = float(evm_vec[EVM_RMS_MAX_PCT])
                print("RESULTS - FREQ ERR AVG Hz: {} \t PWR OUT AVG dBm: {} \t EVM RMS AVG PCT: {}".format(
                    res0[iter_idx], res2[iter_idx], res6[iter_idx]))

                iter_idx += 1

            #results_all = np.vstack((res0, res1, res2, res3, res4, res5, res6, res7))
            results_all = np.vstack((res0, res2, res4, res6))
            # Average over the multiple iterations
            results_all = np.mean(results_all, 1)
            # Round to two decimal places
            results_all = [round(x, 2) for x in results_all]

            results_final = copy.copy(gain_table[gain_count - 1])
            results_final.extend(results_all)

            # CSV file
            with open(filename, 'a') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if first_iter:
                    # Write headers
                    #writer.writerow(['Index', 'TotalGain', 'PA1', 'PA2', 'PA3', 'ATTN', 'IAMP', 'PAD',
                    #                 'FREQ_ERR_AVG_HZ', 'FREQ_ERR_MAX_HZ',
                    #                 'PWR_OUT_AVG_DBM', 'PWR_OUT_MAX_DBM',
                    #                 'PWR_MEAN_AVG_DBM', 'PWR_MEAN_MAX_DBM',
                    #                 'EVM_RMS_AVG_PCT', 'EVM_RMS_MAX_PCT'])
                    writer.writerow(['Index', 'TotalGain', 'PA1', 'PA2', 'PA3', 'ATTN', 'IAMP', 'PAD',
                                     'FREQ_ERR_AVG_HZ',
                                     'PWR_OUT_AVG_DBM',
                                     'PWR_MEAN_AVG_DBM',
                                     'EVM_RMS_AVG_PCT'])
                    writer.writerow(results_final)
                    first_iter = False
                else:
                    writer.writerow(results_final)

    print("(1) Stopped specan_meas thread!")
    sys.exit()


def generate_signal(sig_type):

    debug = False

    matlab = True
    anritsu = False
    LTE = False

    # LTE E-UTRA Test Model (currently available: 3-1, 3-2, and 3-3)
    if sig_type == "TM3_1":
        tm_num = '3_1'
    elif sig_type == "TM3_2":
        tm_num = '3_2'
    elif sig_type == "TM3_3":
        tm_num = '3_3'
    else:
        print("Signal Type Not Available")
        sys.exit(0)

    if matlab:
        #dataset = loadmat('./data_in/LTE_FDD/Matlab/LTE_E-TM_waveforms.mat')
        dataset = loadmat('./data_in/3GPP_Matlab/LTE_and_WiFi_TestSignals.mat')

        '''
        # SIGNAL TYPES
        # lte_tm3_2_fdd_5MHz_waveform
        # lte_tm3_2_tdd_5MHz_waveform
        # ofdm_qam4_waveform
        # qam4_waveform
        # wifi_11ag_20MHz_waveform
        # wifi_11j_10MHzPN9_waveform
        # wifi_11p_5MHz_BPSK_100bytes_waveform
        '''
        # tmgrid = dataset['tmgrid_' + tm_num]
        # tmconfig = dataset[]   #dataset['tmconfig_' + tm_num]
        tmwaveform = dataset['lte_tm3_2_fdd_5MHz_waveform']   #dataset['tmwaveform_' + tm_num]

        numSamps = tmwaveform.size
        txSignal = np.empty(numSamps).astype(np.complex64)
        for idx in range(numSamps):
            txSignal[idx] = tmwaveform[idx]

        # PAPR
        txSignal_abs = abs(txSignal)
        PAPR = max(txSignal_abs**2) / np.mean(txSignal_abs**2)
        PAPR_db = 10*np.log10(PAPR)

    elif anritsu:
        ###########################################################
        # FIXME # don't know what the format of the input data is #
        ###########################################################
        data = np.fromfile("./data_in/3GPP_Anritsu/E-TM_3-2_05M.wvd", dtype='<i2', count=-1, sep='')
        numSamps = len(data)//2
        full_iq = np.empty(numSamps).astype(np.complex64)
        count = 0

        for idx in range(0, len(data), 2):
            full_iq[count] = np.complex(data[idx] / 32768.0, data[idx+1] / 32768.0)
            count += 1
        txSignal = full_iq

    elif LTE:
        # LTE signal generated by Rahman? Not sure about the contents of the signal, seems to be QPSK
        # Generate TX signal
        numSamps = 76800
        txSignal = np.empty(numSamps).astype(np.complex64)
        for i in range(numSamps):
            txSignal[i] = np.complex(LTE5_re.lte5i[i] / 32768.0, LTE5_im.lte5q[i] / 32768.0)

    # Scale the Tx vector to +/- 1
    TX_SCALE = 1
    txSignal = TX_SCALE * txSignal / max(abs(txSignal))

    # PAPR
    txSignal_abs = abs(txSignal)
    PAPR = max(txSignal_abs ** 2) / np.mean(txSignal_abs ** 2)
    PAPR_db = 10 * np.log10(PAPR)

    if debug:
        plt.figure(100)
        plt.plot(np.real(txSignal))
        plt.show(block=False)
        plt.figure(101)
        plt.plot(abs(txSignal))
        plt.show()
    return txSignal


def signal_handler(sdr, txStream, rxStream, signal, frame):
    global running
    running = False
    print("Exiting Program. Cleanup Streams")
    if txStream is not None:
        sdr.deactivateStream(txStream)
        sdr.deactivateStream(rxStream)
        sdr.closeStream(txStream)
        sdr.closeStream(rxStream)
    sys.exit(0)


#########################################
#                 Main                  #
#########################################
def main():

    parser = OptionParser()
    parser.add_option("--rate", type="float", dest="rate", help="Tx sample rate", default=7.68e6)  # for 5MHz rate= 7.68
    parser.add_option("--bw", type="float", dest="bw", help="Bandwidth", default=5e6)  # 5MHz
    parser.add_option("--num_ant", type="string", dest="num_ant", help="Optional Tx antenna", default=1)
    parser.add_option("--freq", type="string", dest="freq", help="Tx RF freq band (HI=3.6GHz, LO=2.5GHz)", default="HI")
    parser.add_option("--gain", type="float", dest="gain", help="Tx Gain (dB)", default=80.0)
    parser.add_option("--serial", type="string", dest="serial", help="serial number of the device",
                      default="RF3E000059")
    parser.add_option("--sig_type", type="string", dest="sig_type", help="Signal Type", default="TM3_2")
    parser.add_option("--gain_sweep", action="store_true", dest="gain_sweep", help="Gain Sweep", default=False)
    parser.add_option("--gain_table", action="store_true", dest="gain_table_en", help="Gain Table Verify", default=False)
    parser.add_option("--specan_remote", action="store_true", dest="specan_remote", help="Connect to remote spectrum "
                                                                                         "analyzer for "
                                                                                         "characterization",
                      default=False)
    (options, args) = parser.parse_args()

    if options.gain_sweep and options.gain_table_en:
        print("Can't select both gain sweep and gain table, must disable one.")

    # Display parameters
    print("\n")
    print("========== TX PARAMETERS =========")
    print("Transmitting {} signal from board {}".format(options.sig_type, options.serial))
    print("Sample Rate (sps): {}".format(options.rate))
    print("# of Antennas: {}".format(options.num_ant))
    print("Frequency (Hz): {}".format(options.freq))
    print("Gain Sweep? {}".format(options.gain_sweep))
    print("Spectrum Analyzer? {}".format(options.specan_remote))
    print("==================================")
    print("\n")

    # Create gain table for gain sweep
    if options.freq == "LO":
        fre_str = "LO"
        freq = 2.5e9
        PA1 = [14]                    # on/off
        PA2 = [0]  #[0, 17]           # bypass - disable PA2 for safety
        PA3 = [31.5]                  # on/off
        ATTN = [-18, -12, -6, 0]      #
        IAMP = list(range(-12, 12+1)) # MAX at 12
        PAD = list(range(0, 52+1))    # MAX at 52
        # IAMP = list(range(1, 3+1))  # CAP at 3
        # PAD = list(range(31, 33+1))    # CAP at 33

    elif options.freq == "HI":
        fre_str = "HI"
        freq = 3.6e9
        PA1 = [13.7]                  # on/off
        PA2 = [0]  #[0, 14]           # bypass - disable PA2 for safety
        PA3 = [31]                    # on/off
        ATTN = [-6, 0]   #[-18, -12, -6, 0]      #
        IAMP = list(range(0, 12+1))  #list(range(-12, 12+1)) # MAX at 12
        PAD = list(range(42, 52+1))  #list(range(0, 52+1))    # MAX at 52
    else:
        print("")
        sys.exit(0)

    # WITH CURRENT VALUES, IT SHOULD BE A TOTAL OF 5300 VALUES
    if options.gain_table_en:
        if options.freq == "LO":
            table_filename = './GainTablesOct2021/gain_table_TX_LO.csv'
        elif options.freq == "HI":
            table_filename = './GainTablesOct2021/gain_table_TX_HI.csv'
        else:
            print("Set Frequency!")
            sys.exit(0)

        gain_table = []
        with open(table_filename, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # Format: TOTAL_GAIN, PA1, PA3, PA2, ATTN, PAD, IAMP
                total = float(row[0])
                PA1 = float(row[1])
                PA2 = float(row[3])
                PA3 = float(row[2])
                ATTN = float(row[4])
                PAD = float(row[5])
                IAMP = float(row[6])
                gain_table.append([total, PA1, PA2, PA3, ATTN, IAMP, PAD])
    else:
        count = 0
        gain_table = []
        for idx0 in PA1:
            for idx1 in PA2:
                for idx2 in PA3:
                    for idx3 in ATTN:
                        for idx4 in IAMP:
                            for idx5 in PAD:
                                total = sum([idx0, idx1, idx2, idx3, idx4, idx5])
                                gain_table.append([count, total, idx0, idx1, idx2, idx3, idx4, idx5])
                                count += 1

    print("TABLE LENGTH: {}".format(len(gain_table)))
    if options.specan_remote:
        # Initialize Anritsu Remote controller
        specan = AnritsuRemote(freq, options.bw, options.sig_type)
        # Initialize DL LTE 3G APP
        specan.app_dl_lte3G()
    else:
        specan = []

    tx_char_app(
        rate=options.rate,
        num_ant=options.num_ant,
        freq=freq,
        freq_str=fre_str,
        gain=options.gain,
        serial=options.serial,
        sig_type=options.sig_type,
        gain_sweep=options.gain_sweep,
        specan=specan,
        gain_table=gain_table,
        specan_remote=options.specan_remote,
        gain_table_en=options.gain_table_en,
    )


if __name__ == '__main__':
    main()
