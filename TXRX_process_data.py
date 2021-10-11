#!/usr/bin/python
"""
    process_rx_gain_char.py

    Script does the following:
    1) Reads noise data from RX characterization and plots
       it, if plotting is enabled
    2) Reads signal data from RX characterization and plots
       it, if plotting is enabled
    3) Generates gain table
---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
---------------------------------------------------------------------
"""

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from datetime import date
import math

plt.style.use('ggplot')  # customize your plots style


################
#    GLOBAL    #
################
PLOT_SIGNAL = False
PLOT_SNR = False
PLOT_EVM = False


def rx_signal_new(path, freq, board_serials, sig_lev):
    """

    """
    global PLOT_SIGNAL
    # Read data
    num_boards = len(board_serials)
    rssi_lms7 = []
    rssi_fpga = []
    pwr_time = []
    pwr_freq = []

    for boardIdx in range(num_boards):
        file = path + "rx_vs_rssi_" + freq + "_txPwr" + sig_lev + "dBm_extAttn15dB_" + board_serials[boardIdx] + ".csv"

        df_signal = pd.read_csv(file)
        header = list(df_signal.columns.values)

        # Get data
        LNA = list(df_signal['LNA'])
        TIA = list(df_signal['TIA'])
        PGA = list(df_signal['PGA'])
        LNA1 = list(df_signal['LNA1'])
        LNA2 = list(df_signal['LNA2'])
        ATTN = list(df_signal['ATTN'])

        tmp_lms7 = list(df_signal['RSSI'].values.astype(float))
        tmp_fpga = list(df_signal['RSSI_FPGA'].values.astype(float))
        tmp_pwrtime = list(df_signal['PWR_TIME (LIN)'].values.astype(float))
        tmp_pwrfreq = list(df_signal['PWR_FREQ (LIN)'].values.astype(float))

        rssi_lms7.append(tmp_lms7)
        rssi_fpga.append(tmp_fpga)
        pwr_time.append(tmp_pwrtime)
        pwr_freq.append(tmp_pwrfreq)

        zero_vals = np.where(np.array(tmp_lms7) == 0.0)[0]
        if any(zero_vals):
            print("BAD SIG ENTRIES BOARD " + board_serials[boardIdx] + " SIGNAL " + sig_lev + "dBm:")
            for idx, val in enumerate(zero_vals):
                print("AT LNA {} \t TIA {} \t PGA {} \t LNA1 {} \t LNA2 {} \t ATTN {}".format(LNA[val], TIA[val], PGA[val], LNA1[val], LNA2[val], ATTN[val]))

    rssi_all = rssi_lms7
    avg_rssi = np.mean(rssi_lms7, 0)
    std_rssi = np.std(rssi_lms7, 0)

    gain_settings = dict({'LNA': LNA, 'TIA': TIA, 'PGA': PGA, 'LNA1': LNA1, 'LNA2': LNA2, 'ATTN': ATTN})

    # Total gain
    total_gain = [sum(x) for x in zip(LNA, TIA, PGA, LNA1, LNA2, ATTN)]

    # ======= PLOT =======
    if PLOT_SIGNAL:
        color_vec = ['red', 'blue', 'green', 'magenta', 'black']

        fig = plt.figure()
        ax1 = fig.add_subplot(5, 1, 1)

        ax1.grid(True)
        ax1.set_title("RSSI (Signal)" + sig_lev + " dBm")
        ax1.set_ylabel("Digital RSSI LMS7")
        for idx, ser in enumerate(board_serials):
            ax1.plot(rssi_lms7[idx], '--o', color=color_vec[idx], alpha=0.7, label=ser)
        ax1.legend(fontsize=10)

        ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
        ax2.grid(True)
        ax2.set_title("RSSI (Signal)" + sig_lev + " dBm")
        ax2.set_ylabel("Digital RSSI FPGA")
        for idx, ser in enumerate(board_serials):
            ax2.plot(rssi_fpga[idx], '--o', color=color_vec[idx], alpha=0.7, label=ser)
        ax2.legend(fontsize=10)

        ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
        ax3.grid(True)
        ax3.set_title("Avg RSSI across Boards (Signal)" + sig_lev + " dBm")
        ax3.set_ylabel("Digital RSSI")
        ax3.plot(avg_rssi, 'o', color='blue', alpha=1, label='RSSI (MEAN)')
        ax3.fill_between(range(len(avg_rssi)), avg_rssi+std_rssi, avg_rssi-std_rssi, color='#539caf', alpha=0.6, label='RSSI (STD)')
        ax3.legend(fontsize=10)

        ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
        ax4.grid(True)
        ax4.set_title("LMS7 Gains")
        ax4.set_ylabel("Gain (dB)")
        ax4.plot(LNA, label='LNA')
        ax4.plot(TIA, label='TIA')
        ax4.plot(PGA, label='PGA')
        ax4.plot(total_gain, '--o', color='blue', alpha=0.7, label='Total Gain')
        ax4.legend(fontsize=10)

        ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)
        ax5.grid(True)
        ax5.set_title("CBRS FE Gains")
        ax5.set_ylabel("Gain (dB)")
        ax5.plot(LNA1, label='LNA1')
        ax5.plot(LNA2, label='LNA2')
        ax5.plot(ATTN, '--', label='ATTN')
        ax5.legend(fontsize=10)
        plt.show(block=False)

    # Average across boards, and full matrix for all boards
    return gain_settings, avg_rssi, rssi_all


def rx_evm(path, freq, board_serials, sig_lev):
    """
    EVM data collected on Jan 2020
    """
    global PLOT_EVM

    # Read data
    num_boards = len(board_serials)
    evm_pct = []
    evm_snr = []

    for boardIdx in range(num_boards):
        file = path + "rx_vs_evm_" + freq + "_txPwr" + sig_lev + "dBm_extAttn15dB_" + board_serials[boardIdx] + ".csv"
        df_signal = pd.read_csv(file)
        header = list(df_signal.columns.values)

        # Get data
        LNA = list(df_signal['LNA'])
        TIA = list(df_signal['TIA'])
        PGA = list(df_signal['PGA'])
        LNA1 = list(df_signal['LNA1'])
        LNA2 = list(df_signal['LNA2'])
        ATTN = list(df_signal['ATTN'])

        tmp_evm_pct = list(df_signal['EVM_PCT'].values.astype(float))
        tmp_evm_snr = list(df_signal['EVM_SNR'].values.astype(float))

        for idx in range(len(tmp_evm_pct)):
            if tmp_evm_pct[idx] < 0 or tmp_evm_pct[idx] > 100:
                tmp_evm_pct[idx] = float('nan')

        evm_pct.append(tmp_evm_pct)
        evm_snr.append(tmp_evm_snr)

    evm_all = evm_pct
    snr_all = evm_snr
    avg_evm = np.mean(evm_pct, 0)
    std_evm = np.std(evm_pct, 0)
    avg_snr = np.mean(snr_all, 0)
    std_snr = np.std(snr_all, 0)

    gain_settings = dict({'LNA': LNA, 'TIA': TIA, 'PGA': PGA, 'LNA1': LNA1, 'LNA2': LNA2, 'ATTN': ATTN})

    # Total gain
    total_gain = [sum(x) for x in zip(LNA, TIA, PGA, LNA1, LNA2, ATTN)]

    # ======= PLOT =======
    if PLOT_EVM:
        color_vec = ['red', 'blue', 'green', 'magenta', 'black']

        fig = plt.figure()
        ax1 = fig.add_subplot(5, 1, 1)

        ax1.grid(True)
        ax1.set_title("EVM FOR EACH BOARD" + sig_lev + " dBm")
        ax1.set_ylabel("EVM (%)")
        for idx, ser in enumerate(board_serials):
            ax1.plot(evm_all[idx], '--o', color=color_vec[idx], alpha=0.7, label=ser)
        ax1.legend(fontsize=10)

        ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
        ax2.grid(True)
        ax2.set_title("AVG EVM ACROSS BOARDS" + sig_lev + " dBm")
        ax2.set_ylabel("EVM (%)")
        ax2.plot(avg_evm, 'o', color='blue', alpha=1, label='EVM (MEAN)')
        ax2.fill_between(range(len(avg_evm)), avg_evm + std_evm, avg_evm - std_evm, color='#539caf', alpha=0.6, label='RSSI (STD)')
        ax2.legend(fontsize=10)

        ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
        ax3.grid(True)
        ax3.set_title("AVG SNR ACROSS BOARDS" + sig_lev + " dBm")
        ax3.set_ylabel("SNR (dB)")
        ax3.plot(avg_snr, 'o', color='blue', alpha=1, label='SNR (MEAN)')
        ax3.fill_between(range(len(avg_snr)), avg_snr + std_snr, avg_snr - std_snr, color='#539caf', alpha=0.6, label='RSSI (STD)')
        ax3.legend(fontsize=10)

        ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
        ax4.grid(True)
        ax4.set_title("LMS7 Gains")
        ax4.set_ylabel("Gain (dB)")
        ax4.plot(LNA, label='LNA')
        ax4.plot(TIA, label='TIA')
        ax4.plot(PGA, label='PGA')
        ax4.plot(total_gain, '--o', color='blue', alpha=0.7, label='Total Gain')
        ax4.legend(fontsize=10)

        ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)
        ax5.grid(True)
        ax5.set_title("CBRS FE Gains")
        ax5.set_ylabel("Gain (dB)")
        ax5.plot(LNA1, label='LNA1')
        ax5.plot(LNA2, label='LNA2')
        ax5.plot(ATTN, '--', label='ATTN')
        ax5.legend(fontsize=10)
        plt.show(block=False)

    # Average across boards, and full matrix for all boards
    return gain_settings, avg_evm, evm_all


def snr(signal_mat, noise_mat, sig_lev, max_sig_level, board_serials):
    """

    """
    global PLOT_SNR
    signal_mat = np.asarray(signal_mat)
    noise_mat = np.asarray(noise_mat)

    mat_shape = signal_mat.shape
    snr_all = []
    for idx in range(mat_shape[0]):
        signal = signal_mat[idx]
        noise = noise_mat[idx]

        # SNR
        max_rssi = 90100    # Used by LimeSuite as approximation
        signal_vec_dB = 20 * np.log10(signal / max_rssi)
        noise_vec_dB = 20 * np.log10(noise / max_rssi)
        sig_lev_int = int(sig_lev)
        # Normalize to max pwr level used
        new_snr = signal_vec_dB + (abs(sig_lev_int) - abs(max_sig_level)) - noise_vec_dB

        # Unfortunately we got some bad entries, remove them... need to fix this when collecting data
        # replace with zero SNR
        for idx, val in enumerate(new_snr):
            if np.isinf(val) or np.isnan(val):
                print("BAD SNR AT IDX {}, SNR {}".format(idx, val))
                new_snr[idx] = 0.0

        snr_all.append(new_snr)

    avg_snr = np.mean(snr_all, 0)
    std_snr = np.std(snr_all, 0)
    snr_vec = avg_snr

    # ======= PLOT =======
    if PLOT_SNR:
        color_vec = ['red', 'blue', 'green', 'magenta', 'black']

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.grid(True)
        ax1.set_title("Per Board SNR " + sig_lev + " dBm")
        ax1.set_ylabel("SNR dBFS")
        for idx, ser in enumerate(board_serials):
            ax1.plot(snr_all[idx], '--o', color=color_vec[idx], alpha=0.7, label=ser)
        ax1.set_ylim([0, 100])
        ax1.legend(fontsize=10)

        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.grid(True)
        ax2.set_title("Avg SNR across Boards " + sig_lev + " dBm")
        ax2.set_ylabel("SNR dBFS")
        ax2.plot(avg_snr, 'o', color='blue', alpha=1, label='SNR (MEAN)')
        ax2.fill_between(range(len(avg_snr)), avg_snr+std_snr, avg_snr-std_snr, color='#539caf', alpha=0.6, label='SNR (STD)')
        ax2.set_ylim([0, 100])
        ax2.legend(fontsize=10)
        plt.show(block=False)

    return snr_vec, snr_all


def select_snr_sections(snr, freq):
    """
    We use different signal levels as input during characterization. We do this to avoid using SNR from saturated
    samples. In data collected for 2.5GHz on August 2019 we used the following TX power [-50, -60, -70, -80] dBm.
    By looking at the RSSI plots for each case we select the gain setting ranges where the signal is "decent" (not too
    low to be buried in the noise, but not too high to cause clipping)

    New data collected on Jan 2020. 2.5GHz [-50, -60, -70, -80, -90] dBm and 3.6GHz [-20, -30, -40, -50, -60] dBm
    Noise in files with -999 tag
    """

    # Total of 8640 samples (15 LNA, 3 TIA, 32 PGA, 1 LNA1, 2 LNA2, 3 ATTN) NOTE: 4 ATTN in old, 3 in new
    # We decided to split into segments of 1440 samples each (Where TIA changes, i.e., LNA*PGA = 45*32 = 480)
    samps_per_seg = 1440
    segments = range(0, 8640, samps_per_seg)  # 18 sections
    # sections_half = range(0, 11520//2, 480)  # 12 sections (for datasets where we only collected first half)

    if freq == 'LO':
        section1 = list(range(segments[0:7][0], segments[0:7][-1] + samps_per_seg))
        section2 = list(range(segments[7:13][0], segments[7:13][-1] + samps_per_seg))
        section3 = list(range(segments[13:20][0], segments[13:20][-1] + samps_per_seg))
        section4 = list(range(segments[20:24][0], segments[20:24][-1] + samps_per_seg))
    elif freq == 'HI':
        section1 = list(range(segments[0:1][0], segments[0:1][-1] + samps_per_seg))
        section2 = list(range(segments[1:2][0], segments[1:2][-1] + samps_per_seg))
        section3 = list(range(segments[2:4][0], segments[2:4][-1] + samps_per_seg))
        section4 = list(range(segments[4:5][0], segments[4:5][-1] + samps_per_seg))
        section5 = list(range(segments[5:6][0], segments[5:6][-1] + samps_per_seg))

    snr_final = []
    snr_final.extend(snr[0][section1])
    snr_final.extend(snr[1][section2])
    snr_final.extend(snr[2][section3])
    snr_final.extend(snr[3][section4])
    snr_final.extend(snr[4][section5])

    # Plot final SNR
    plt.figure(100)
    plt.grid(True)
    plt.plot(snr_final, '--o', alpha=0.7, label='SNR final')
    plt.show()

    return snr_final


def generate_rx_gain_table(mode, data_vec, gain_settings):
    """
    Generate gain table by computing SNR from signal and noise vectors and
    sort by total gain. We have two modes:
    a) Min EVM: Select gain setting that minimizes EVM
    b) Max SNR: For each amount of total gain, pick rows that maximize
                SNR (gain settings that maximize SNR)
    """

    # Gain Settings
    LNA = [int(i) for i in gain_settings['LNA']]        # [0:1:30]
    TIA = [int(i) for i in gain_settings['TIA']]        # [0, 3, 9, 12]
    PGA = [int(i) for i in gain_settings['PGA']]        # [-12:1:19] ... 32dB control range according to datasheet
    LNA1 = [int(i) for i in gain_settings['LNA1']]      # [0, 33]
    LNA2 = [int(i) for i in gain_settings['LNA2']]      # [0, 17]
    ATTN = [int(i) for i in gain_settings['ATTN']]      # [-18, -12, -6, 0]

    # Total gain
    total_gain = [int(sum(x)) for x in zip(LNA, TIA, PGA, LNA1, LNA2, ATTN)]

    # Create data matrix
    dataMatrix = np.column_stack((total_gain, data_vec, LNA, TIA, PGA, LNA1, LNA2, ATTN))
    entriesPerVec = len(dataMatrix[0])

    # Sort by "totalGain"
    # sort array with regards to Nth column
    TOT_GAIN_COL = 0
    DATA_COL = 1
    dataMatrix = dataMatrix[dataMatrix[:, TOT_GAIN_COL].argsort()]
    print(dataMatrix)

    # Keep entries with maximum SNR OR minimum EVM (depending on mode)
    uniqueGainVals = np.unique(dataMatrix[:, TOT_GAIN_COL])
    gainTable = np.zeros((len(uniqueGainVals), entriesPerVec))
    data_best = []
    for idx, gain in enumerate(uniqueGainVals):
        currGainIdx = gain == dataMatrix[:, TOT_GAIN_COL]   # find rows with same amount of total gain
        gainVecs = dataMatrix[currGainIdx, :]               # get data for those entries

        if mode == 'SNR':
            maxIdx = np.argmax(gainVecs[:, DATA_COL])           # find index of entry with max SNR
            maxVec = gainVecs[maxIdx, :]                        # grab entry by using that index
            gainTable[idx, ] = maxVec                           # record vector
        elif mode == 'EVM':
            minIdx = np.argmin(gainVecs[:, DATA_COL])           # find index of entry with max SNR
            minVec = gainVecs[minIdx, :]                        # grab entry by using that index
            gainTable[idx, ] = minVec                           # record vector

        data_best.append(gainTable[idx, 1])

    dt = date.today().strftime("%m_%d_%y")
    table_file = "gainTable_CBRS_" + dt + ".csv"
    # Format: total_gain, snr, LNA, TIA, PGA, LNA1, LNA2, ATTN
    np.savetxt(table_file, gainTable, delimiter=",")

    # PRINT GAIN TABLE - tables will go into FW (/fw/iris030/soapy_iris/gain_tables.hpp)
    # Print it in C++ table format. Will be used by another script to convert to actual gain settings to form the table
    # that will be hardcoded into the FPGA
    # dataMatrix format: [total_gain, snr/evm, LNA, TIA, PGA, LNA1, LNA2, ATTN]
    # gainTable  format: [total_gain, ATTN, LNA1, LNA2, LNA, TIA, PGA]
    print("int gainTable[NUM_GAIN_LEVELS][7] = {")
    for rowIdx in range(len(gainTable[:, 0])):
        evm_val = round(gainTable[rowIdx, 1], 2)
        print("{\t" + str(int(gainTable[rowIdx, 0])) + " \t , \t " + str(int(gainTable[rowIdx, 7])) + " \t , \t " + str(int(gainTable[rowIdx, 5])) + " \t , \t "
                    + str(int(gainTable[rowIdx, 6])) + " \t , \t " + str(int(gainTable[rowIdx, 2])) + " \t , \t " + str(int(gainTable[rowIdx, 3])) + " \t , \t "
                    + str(int(gainTable[rowIdx, 4])) + " \t }, // \t "
                    + str(evm_val))
    print("};")

    plt.figure()
    plt.plot(data_best)
    plt.grid()
    plt.show()


def generate_tx_gain_table(freq):
    """
     Generate TX table for CBRS board.

     TWO METHODS:

     1) OLD "ARBITRARY" METHOD: Main factor to consider is the noise figure of each component (each amplifier) in TX chain.
     Currently, preference is as follows (from lower to higher noise).
     Prioritize as follows (when possible):
     PA2 -> ATTN -> PAD -> IAMP              (OLD & WRONG: IAMP > PAD > ATTN > PA2)

     2) EVM BASED METHOD: Read file with EVM measurements for all gains. Pick gain combination that minimizes EVM
    """

    new = True

    if new:
        freq_str = freq
        print("Frequency: {}".format(freq_str))
        # EVM BASED METHOD
        # Data frame
        filename = "./data_in/tx_vs_evm_" + freq_str + "_15dB_ExtAtten.csv"
        df_txdata = pd.read_csv(filename)

        # Get data
        # Round total gain values
        total_gain = [math.ceil(x) for x in list(df_txdata['TotalGain'])]

        PA1 = list(df_txdata['PA1'])
        PA2 = list(df_txdata['PA2'])
        PA3 = list(df_txdata['PA3'])
        ATTN = list(df_txdata['ATTN'])
        IAMP = list(df_txdata['IAMP'])
        PAD = list(df_txdata['PAD'])

        R0 = list(df_txdata['FREQ_ERR_AVG_HZ'])
        R1 = list(df_txdata['FREQ_ERR_MAX_HZ'])
        R2 = list(df_txdata['PWR_OUT_AVG_DBM'])
        R3 = list(df_txdata['PWR_OUT_MAX_DBM'])
        R4 = list(df_txdata['PWR_MEAN_AVG_DBM'])
        R5 = list(df_txdata['PWR_MEAN_MAX_DBM'])
        R6 = list(df_txdata['EVM_RMS_AVG_PCT'])
        R7 = list(df_txdata['EVM_RMS_MAX_PCT'])

        external_attn = 15 + 1  # dB  15dB from ext atten + 1 dB loss from cabling

        if freq_str == 'LO':
            # Comment out this piece of code once acknowledged
            print("STOP!!!! if generating a new table, remember we are missing data from 88dB total gain, just add it "
                  "manually to hpp table. You'll have to re-run this table generator without this check but put it back"
                  "in afterwards in case someone has to generate the table again. "
                  "Yeah, annoying but it will keep us from crashing the FW...")
            sys.exit(0)

        best_idx = []
        best_entry = []
        total_gain_unique = np.unique(total_gain)
        print(np.diff(total_gain_unique))
        for idx, gain in enumerate(total_gain_unique):
            # Indices for each gain value
            indexes = [i for i, x in enumerate(total_gain) if x == gain]
            # print("Value: {} \t Indexes: {}".format(gain, indexes))

            # only care about positive valid values
            tmp = (df_txdata['EVM_RMS_AVG_PCT'][indexes] >= 0) & (df_txdata['EVM_RMS_AVG_PCT'][indexes] != -999.00)
            # index of positive and valid values
            valid_idx = tmp.index[tmp[indexes] == True].tolist()

            # Check if any valid index, otherwise simply use the first one, doesn't matter...
            if not valid_idx:
                evm = df_txdata['EVM_RMS_AVG_PCT'][indexes[0]]
                this_idx = indexes[0]
                best_idx.append(this_idx)
                best_entry.append([total_gain[this_idx],
                                   df_txdata['PA1'][this_idx],
                                   df_txdata['PA3'][this_idx],
                                   df_txdata['PA2'][this_idx],
                                   df_txdata['ATTN'][this_idx],
                                   df_txdata['PAD'][this_idx],
                                   df_txdata['IAMP'][this_idx],
                                   evm,
                                   df_txdata['PWR_MEAN_AVG_DBM'][this_idx] + external_attn])
            else:
                # get index of optimal entry
                evm = df_txdata['EVM_RMS_AVG_PCT'][valid_idx].min()
                this_idx = df_txdata['EVM_RMS_AVG_PCT'][valid_idx].idxmin()
                best_idx.append(this_idx)
                best_entry.append([total_gain[this_idx],
                                   df_txdata['PA1'][this_idx],
                                   df_txdata['PA3'][this_idx],
                                   df_txdata['PA2'][this_idx],
                                   df_txdata['ATTN'][this_idx],
                                   df_txdata['PAD'][this_idx],
                                   df_txdata['IAMP'][this_idx],
                                   evm,
                                   df_txdata['PWR_MEAN_AVG_DBM'][this_idx] + external_attn])

        for idx in range(len(best_entry)):
            # [tot_gain, pa1, pa3, pa2, attn, pad, iamp]
            print("{{ {:0.2f}, \t {}, \t {}, \t {}, \t {}, \t {}, \t {} }},  // EVM: {}  TXPWR: {:0.2f}".format(
                                                                                    best_entry[idx][0],
                                                                                    best_entry[idx][1],
                                                                                    best_entry[idx][2],
                                                                                    best_entry[idx][3],
                                                                                    best_entry[idx][4],
                                                                                    best_entry[idx][5],
                                                                                    best_entry[idx][6],
                                                                                    best_entry[idx][7],
                                                                                    best_entry[idx][8]))
    else:
        # ARBITRARY METHOD
        # CBRS Gains
        if freq == "LO":
            PA1 = [14]        # on/off
            PA2 = [0, 17]   # bypass
            PA3 = [31.5]      # on/off
            ATTN = [-18, -12, -6]  # CAP, 0]
        elif freq == "HI":
            PA1 = [13.7]        # on/off
            PA2 = [0, 14]     # bypass
            PA3 = [31]          # on/off
            ATTN = [-18, -12, -6]  # CAP, 0]
        else:
            print("INVALID FREQ")
            return

        # LMS7 Gains
        IAMP = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]  # CAP, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # According to skylark, there's been some non-linearities above PAD=~42
        PAD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]  #CAP , 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

        min_gain = round(min(PA1) + min(PA2) + min(PA3) + min(ATTN) + min(IAMP) + min(PAD))
        max_gain = round(max(PA1) + max(PA2) + max(PA3) + max(ATTN) + max(IAMP) + max(PAD))
        total_gain_container = list(range(min_gain, max_gain+1))
        print("LENGTH: {}".format(len(total_gain_container)))
        num_stages = 7  # totalGain, pa1, pa2, p3, attn, pad, iamp
        gain_structure = np.empty((len(total_gain_container), num_stages))
        counter = 0
        tot_gain_all = []
        for idx1, pa2 in enumerate(PA2):
            for idx2, pad in enumerate(PAD):
                for idx3, iamp in enumerate(IAMP):
                    for idx4, attn in enumerate(ATTN):
                        for idx5, pa3 in enumerate(PA3):
                            for idx6, pa1 in enumerate(PA1):
                                tot_gain = math.ceil(pa1 + pa3 + pa2 + attn + pad + iamp)
                                tot_gain_all.append(tot_gain)

                                # Check if value has already been added
                                try:
                                    gain_structure_list = gain_structure[:, 0].tolist()
                                    found_idx = gain_structure_list.index(tot_gain)
                                except ValueError:
                                    gain_structure[counter, :] = [tot_gain, pa1, pa3, pa2, attn, pad, iamp]
                                    # print("{:0.2f}, {}, {}, {}, {}, {}, {}".format(tot_gain, pa1, pa3, pa2, attn, pad, iamp))
                                    counter += 1
                                    if counter == 51:
                                        stop = 1
                                    if counter >= len(total_gain_container):
                                        gain_structure = gain_structure[gain_structure[:, 0].argsort()]
                                        for idxtab in range(len(gain_structure[:, 0])):
                                            print("{{ {:0.2f}, \t {}, \t {}, \t {}, \t {}, \t {}, \t {} }},".format(gain_structure[idxtab, 0],
                                                                                                 gain_structure[idxtab, 1],
                                                                                                 gain_structure[idxtab, 2],
                                                                                                 gain_structure[idxtab, 3],
                                                                                                 gain_structure[idxtab, 4],
                                                                                                 gain_structure[idxtab, 5],
                                                                                                 gain_structure[idxtab, 6]))
                                    return


def tx_char():
    filename = "./data_in/TX_Char_out_RF3C000064_16dBExtAtten.csv"
    df_txdata = pd.read_csv(filename)

    # Get data
    PAD = list(df_txdata['PAD'])
    IAMP = list(df_txdata['IAMP'])
    PA1 = list(df_txdata['PA1'])
    PA2 = list(df_txdata['PA2'])
    PA3 = list(df_txdata['PA3'])
    ATTN = list(df_txdata['ATTN'])

    tx_pwr_dBm = df_txdata['AVG_CH_PWR_dBm'] + 16  # 16 dB external attenuation added (15dB atten + 1dB cable loss)

    # Total gain
    total_gain = [sum(x) for x in zip(PAD, IAMP, PA1, PA2, PA3, ATTN)]

    fig = plt.figure(1)
    ax1 = fig.add_subplot(5, 1, 1)
    color = 'red'
    ax1.grid(True)
    ax1.set_title("Tx Pwr and Aggregate Gain")
    ax1.set_ylabel("Tx Pwr (dBm)")
    ax1.plot(tx_pwr_dBm, '--o', color=color, alpha=0.7, label='TxPwr (dBm)')
    ax1.tick_params(axis='y', labelcolor=color)
    color = 'blue'
    ax11 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax11.set_ylabel("Total Gain")
    ax11.plot(total_gain, '--o', color=color, alpha=0.05, label='TotalGain')
    ax11.tick_params(axis='y', labelcolor=color)

    ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
    color = 'red'
    ax2.grid(True)
    ax2.set_title("Tx Pwr")
    ax2.set_ylabel("Tx Pwr (dBm)")
    ax2.plot(tx_pwr_dBm, '--o', color=color, alpha=0.7, label='TxPwr (dBm)')
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
    color = 'blue'
    ax3.grid(True)
    ax3.set_title("Aggregate Gain")
    ax3.set_ylabel("Total Gain")
    ax3.plot(total_gain, '--o', color=color, alpha=0.7, label='TotalGain')
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
    ax4.grid(True)
    ax4.set_title("LMS7 Gains")
    ax4.plot(PAD, alpha=1, label='PAD')
    ax4.plot(IAMP, alpha=1, label='IAMP')
    ax4.legend(fontsize=10)

    ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)
    ax5.grid(True)
    ax5.set_title("CBRS FE Gains")
    ax5.plot(PA1, alpha=1, label='PA1')
    ax5.plot(PA2, alpha=1, label='PA2')
    ax5.plot(PA3, alpha=1, label='PA3')
    ax5.plot(ATTN, '--', alpha=1, label='ATTN')
    ax5.legend(fontsize=10)

    plt.show()


#########################################
#                  Main                 #
#########################################
def main():

    freq = "LO"  # HI: 3.6GHz, LO: 2.5GHz

    rx = True
    tx = False

    if rx:
        mode = 'EVM'  # EVM or SNR
        path = "./data_in/EVM_NEW/"

        if mode == 'SNR':
            # Get signal and snr
            if freq == "LO":
                sig_lev = [-50, -60, -70, -80 -90]
                board_serials = ["RF3C000064", "RF3C000025"]   # "RF3C000042" IS BROKEN
            elif freq == "HI":
                sig_lev = [-20, -30, -40, -50, -60]
                board_serials = ["RF3E000375", "RF3E000375"]  # ["RF3E000392", "RF3E000375"]
            else:
                print("INVALID FREQ")
                return

            # All tx gains + noise
            # Noise first -> Noise Floor file: TxPwr -999
            gain_settings, noise_vec, noise_mat = rx_signal_new(path, freq, board_serials, str(-999))
            max_sig_level = sig_lev[0]
            snr_vec_all_sig = []
            for signal in sig_lev:
                _, signal_vec, signal_mat = rx_signal_new(path, freq, board_serials, str(signal))
                snr_vec, snr_all = snr(signal_mat, noise_mat, str(signal), max_sig_level, board_serials)
                snr_vec_all_sig.append(snr_vec)

            snr_final = select_snr_sections(snr_vec_all_sig, freq)
            generate_rx_gain_table(mode, snr_final, gain_settings)

        elif mode == 'EVM':
            # Get signal and snr
            if freq == "LO":
                sig_lev = [-20, -30, -40, -50, -60]
                board_serials = ["RF3E000392", "RF3E000375"]
            elif freq == "HI":
                sig_lev = [-10, -20, -30, -40, -50, -60]
                board_serials = ["RF3E000392", "RF3E000375"]
            else:
                print("INVALID FREQ")
                return

            tmp_sig = []
            for signal in sig_lev:
                gain_settings, evm_vec, evm_mat = rx_evm(path, freq, board_serials, str(signal))
                tmp_board = []
                for board_idx in range(len(board_serials)):
                    curr_vec = evm_mat[board_idx]
                    curr_vec = np.nan_to_num(curr_vec, nan=999.0)
                    tmp_board.append(curr_vec)

                avg_evm_boards = np.amin(tmp_board, axis=0)
                tmp_sig.append(avg_evm_boards)

            evm_vec_min = np.amin(tmp_sig, axis=0)          # Look at min across signal levels
            evm_vec_min_idx = np.argmin(tmp_sig, axis=0)
            generate_rx_gain_table(mode, evm_vec_min, gain_settings)

    if tx:
        generate_tx_gain_table(freq)
        # tx_char()


if __name__ == '__main__':
    main()
