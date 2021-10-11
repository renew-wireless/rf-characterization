#!/usr/bin/python
"""
 ILA_AGC_analysis.py

    Analyze ILA data (Integrated Logic Analyzer)

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 ---------------------------------------------------------------------
"""

import sys
sys.path.append('../IrisUtils/')
sys.path.append('../data_in/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gain_to_gain_setting import *
import struct


def twos_complement(hexstr, bits):
    value = int(hexstr, 16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value


def main():
    old = False
    # Read data from CSV file
    #data = pd.read_csv("./data_in/iladata_60_newFrameTrig.csv", encoding='utf8')
    data = pd.read_csv("./data_in/iladatatest100.csv", encoding='utf8')
    metrics = data.columns
    print("{}".format(metrics))

    # RSSI
    meas_rssi = np.array(data['u_agc_core/meas_rssi[31:0]']).astype(str)
    meas_rssi_vec = []

    # Flags
    nf_flag = np.array(data['u_agc_core/new_frame_flag']).astype(str)
    agc_en = np.array(data['u_agc_core/agc_en']).astype(str)
    override = np.array(data['u_agc_core/gain_override_out']).astype(str)
    corr_flag = np.array(data['u_agc_core/corr_flag']).astype(str)
    agc_trig = np.array(data['u_agc_core/agc_trigger']).astype(str)
    sat_det_good = np.array(data['u_agc_core/sat_det_good']).astype(str)
    final_stage = np.array(data['u_agc_core/final_stage']).astype(str)
    sat_detected = np.array(data['u_agc_core/sat_detected']).astype(str)
    sat_count = np.array(data['u_agc_core/sat_count[5:0]']).astype(str)
    trig_hold = np.array(data['u_agc_core/agc_trigger_hold']).astype(str)
    nf_flag_vec = []
    agc_en_vec = []
    override_vec = []
    corr_flag_vec = []
    agc_trig_vec = []
    sat_det_good_vec = []
    final_stage_vec = []
    sat_detected_vec = []
    sat_count_vec = []
    trig_hold_vec = []

    # DATA
    data_a = np.array(data['u_agc_core/RX_data_a[31:0]']).astype(str)
    data_a_vec_i = []
    data_a_vec_q = []

    # GAIN
    lna = np.array(data['u_agc_core/gain_lna_out[0][3:0]']).astype(str)
    tia = np.array(data['u_agc_core/gain_tia_out[0][1:0]']).astype(str)
    pga = np.array(data['u_agc_core/gain_pga_out[0][11:0]']).astype(str)
    init_gain = np.array(data['u_agc_core/init_gain_reg[7:0]']).astype(str)
    lna_vec = []
    tia_vec = []
    pga_vec = []
    init_gain_vec = []

    # Check algorithm
    test = np.array(data['u_agc_core/test[7:0]']).astype(str)
    rssi_idx = np.array(data['u_agc_core/meas_rssi_index[31:0]']).astype(str)
    cumm_gain_drop = np.array(data['u_agc_core/cummulative_sat_gain_drop[31:0]']).astype(str)
    rssi_diff = np.array(data['u_agc_core/rssi_diff[31:0]']).astype(str)
    adj_gain = np.array(data['u_agc_core/adjusted_gain[7:0]']).astype(str)
    st = np.array(data['u_agc_core/st[1:0]']).astype(str)
    rssi_target = np.array(data['u_agc_core/rssi_target[29:0]']).astype(str)  # For some reason the last ILA had 29:0 instead of 31:0
    fine_tune_stage = np.array(data['u_agc_core/fine_tune_stage[2:0]']).astype(str)
    test_vec = []
    rssi_idx_vec = []
    cumm_gain_drop_vec = []
    rssi_diff_vec = []
    adj_gain_vec = []
    state_vec = []
    rssi_target_vec = []
    fine_tune_stage_vec = []

    # Clock
    data_clk = np.array(data['u_agc_core/DATA_clk']).astype(str)
    data_clk_vec = []

    # WAIT
    wait_cnt_stop = np.array(data['u_agc_core/wait_count_stop']).astype(str)
    wait_cnt_en = np.array(data['u_agc_core/wait_count_en']).astype(str)
    wait_cnt = np.array(data['u_agc_core/wait_count[31:0]']).astype(str)
    wait_thresh = np.array(data['u_agc_core/wait_count_thresh[7:0]']).astype(str)
    wait_cnt_stop_vec = []
    wait_cnt_en_vec = []
    wait_cnt_vec = []
    wait_thresh_vec = []

    # RESET
    st_sys = np.array(data['u_agc_core/st_sys[1:0]']).astype(str)
    data_rst = np.array(data['u_agc_core/DATA_rst']).astype(str)
    sys_data_rst = np.array(data['u_agc_core/SYS_DATA_rst']).astype(str)
    data_agc_rst = np.array(data['u_agc_core/DATA_agc_rst']).astype(str)
    reset_gains = np.array(data['u_agc_core/reset_gains']).astype(str)
    st_sys_vec = []
    data_rst_vec = []
    sys_data_rst_vec = []
    data_agc_rst_vec = []
    reset_gains_vec = []

    for idx, val in enumerate(meas_rssi):
        # RSSI
        meas_rssi_vec.append(int(meas_rssi[idx], 16))

        # Flags
        nf_flag_vec.append(int(nf_flag[idx], 16))
        agc_en_vec.append(int(agc_en[idx], 16))
        override_vec.append(int(override[idx], 16))
        corr_flag_vec.append(int(corr_flag[idx], 16))
        agc_trig_vec.append(int(agc_trig[idx], 16))
        sat_det_good_vec.append(int(sat_det_good[idx], 16))
        final_stage_vec.append(int(final_stage[idx], 16))
        sat_detected_vec.append(int(sat_detected[idx], 16))
        sat_count_vec.append(int(sat_count[idx], 16))
        trig_hold_vec.append(int(trig_hold[idx], 16))

        # DATA
        data_a_twos = twos_complement(data_a[idx][0:4], 16)  # I
        data_a_vec_i.append(data_a_twos / (2**15))
        data_a_twos = twos_complement(data_a[idx][4:8], 16)  # Q
        data_a_vec_q.append(data_a_twos / (2**15))

        # GAIN
        lna_tmp = int(lna[idx], 16)
        tia_tmp = int(tia[idx], 16)
        pga_tmp = int(format(int(pga[idx], 16), '#014b')[2:2+5], 2)   # #014b: pad with zeros for 14 characters including 0b
        lna_vec.append(get_lna_register(lna_tmp))
        tia_vec.append(get_tia_register(tia_tmp))
        pga_vec.append(get_pga_register(pga_tmp))
        init_gain_vec.append(int(init_gain[idx], 16))

        # Check algorithm
        test_vec.append(int(test[idx], 16))
        rssi_idx_vec.append(int(rssi_idx[idx], 16))
        cumm_gain_drop_vec.append(twos_complement(cumm_gain_drop[idx], 32))
        rssi_diff_vec.append(twos_complement(rssi_diff[idx], 32))
        adj_gain_vec.append(twos_complement(adj_gain[idx], 8))
        state_vec.append(int(st[idx], 16))
        rssi_target_vec.append(int(rssi_target[idx], 16))
        fine_tune_stage_vec.append(int(fine_tune_stage[idx], 16))

        # Clock
        data_clk_vec.append(int(data_clk[idx], 16))

        # WAIT
        wait_cnt_stop_vec.append(int(wait_cnt_stop[idx], 16))
        wait_cnt_en_vec.append(int(wait_cnt_en[idx], 16))
        wait_cnt_vec.append(int(wait_cnt[idx], 16))
        wait_thresh_vec.append(int(wait_thresh[idx], 16))

        # RESET
        st_sys_vec.append(int(st_sys[idx], 16))
        data_rst_vec.append(int(data_rst[idx], 16))
        sys_data_rst_vec.append(int(sys_data_rst[idx], 16))
        data_agc_rst_vec.append(int(data_agc_rst[idx], 16))
        reset_gains_vec.append(int(reset_gains[idx], 16))

    # PLOTTER
    fig = plt.figure(1)
    # RSSI
    ax1 = fig.add_subplot(8, 1, 1)
    ax1.plot(meas_rssi_vec, '--r', label='Meas RSSI')
    ax1.set_ylabel("RSSI")
    ax1.legend(fontsize=10)
    # Flags
    ax2 = fig.add_subplot(8, 1, 2, sharex=ax1)
    ax2.plot(nf_flag_vec, 'b', label='New Frame Flag')
    ax2.plot(agc_en_vec, 'r', label='AGC Enable')
    ax2.plot(override_vec, 'k', label='Gain Override')
    ax2.plot(corr_flag_vec, 'g', label='Corr Flag')
    ax2.plot(agc_trig_vec, 'm', label='AGC Trigger')
    ax2.plot(sat_det_good_vec, '--sb', label='Sat Det Good')
    ax2.plot(final_stage_vec, '--r', label='Final Stage')
    ax2.plot(sat_detected_vec, '--k', label='Sat Detected')
    ax2.plot(sat_count_vec, '--g', label='SatCount')
    ax2.plot(trig_hold_vec, '--m', label='Trig Hold')
    ax2.set_ylim(-0.2, 1.2)
    ax2.legend(fontsize=10)
    # Data
    ax3 = fig.add_subplot(8, 1, 3, sharex=ax1)
    ax3.plot(np.real(data_a_vec_i), 'r', label='Data I')
    ax3.plot(np.imag(data_a_vec_q), '--b', label='Data Q')
    ax3.set_ylim(-1.2, 1.2)
    ax3.legend(fontsize=10)
    # Gain
    ax4 = fig.add_subplot(8, 1, 4, sharex=ax1)
    ax4.plot(lna_vec, 'r', label='LNA')
    ax4.plot(tia_vec, 'b', label='TIA')
    ax4.plot(pga_vec, 'g', label='PGA')
    ax4.plot(init_gain_vec, '--m', label='INIT')
    ax4.set_ylabel("Gain (dB)")
    ax4.set_ylim(-15, 109)
    ax4.legend(fontsize=10)
    # Algorithm
    ax5 = fig.add_subplot(8, 1, 5, sharex=ax1)
    ax5.plot(test_vec, '--g', label='Test')
    ax5.plot(rssi_idx_vec, 'b', label='RSSI Idx')
    ax5.plot(cumm_gain_drop_vec, 'm', label='Cummulative Gain Drop')
    ax5.plot(rssi_diff_vec, 'r', label='RSSI Diff')
    ax5.plot(state_vec, '-sr', label='State')
    ax5.plot(rssi_target_vec, 'g', label='RSSI Target')
    ax5.plot(fine_tune_stage_vec, '--b', label='FineTune Stage')
    ax55 = ax5.twinx()
    ax55.plot(adj_gain_vec, '--r', label='Adj Gain')
    ax5.legend(fontsize=10)
    # Clock
    ax6 = fig.add_subplot(8, 1, 6, sharex=ax1)
    ax6.plot(data_clk_vec, 'b', label='Data Clock')
    ax6.set_ylim(-0.2, 1.2)
    ax6.legend(fontsize=10)
    # Wait
    ax7 = fig.add_subplot(8, 1, 7, sharex=ax1)
    ax7.plot(wait_cnt_stop_vec, 'r', label='Wait Stop')
    ax7.plot(wait_cnt_en_vec, 'b', label='Wait Count Enable')
    ax77 = ax7.twinx()
    ax77.plot(wait_cnt_vec, 'm', label='Wait Count')
    ax77.plot(wait_thresh_vec, '--m', label='Wait Count Thresh')
    ax77.set_ylabel('Wait Count and Thresh', color='m')
    ax7.set_ylim(-0.2, 1.2)
    ax7.legend(fontsize=10)
    # Reset
    ax8 = fig.add_subplot(8, 1, 8, sharex=ax1)
    ax8.plot(st_sys_vec, 'r', label='State SYS')
    ax8.plot(data_rst_vec, 'b', label='Data Rst')
    ax8.plot(sys_data_rst_vec, 'm', label='Sys Data Rst')
    ax8.plot(data_agc_rst_vec, 'g', label='Data AGC Rst')
    ax8.plot(reset_gains_vec, '--r', label='Reset Gains')
    ax8.set_ylim(-0.2, 4)
    ax8.legend(fontsize=10)
    # Show
    plt.show()


if __name__ == '__main__': 
    main()
