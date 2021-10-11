#!/usr/bin/python3
"""
 gain_to_gain_setting.py

 Given a gain value, retrieve the gain setting that maximizes SNR
 for that gain value

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
---------------------------------------------------------------------
"""

import numpy as np
import pandas as pd


def list_all_gain_combinations():
    """
    
    """
    PAD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
    IAMP = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    PA1 = [14]  # [13.7]
    PA2 = [0, 14]
    PA3 = [31]
    ATTN = [-18, -12, -6]  # , 0]  # ignore 0

    for pa1_idx, pa1 in enumerate(PA1):
        for pa3_idx, pa3 in enumerate(PA3):
            for pa2_idx, pa2 in enumerate(PA2):
                for attn_idx, attn in enumerate(ATTN):
                    for pad_idx, pad in enumerate(PAD):
                        for iamp_idx, iamp in enumerate(IAMP):
                            print("total gain: {}".format(pa1+pa3+pa2+attn+pad+iamp))


def gain_to_gain_setting(gain):
    """
    Find best gain setting for a desired gain value. Best==setting that maximizes SNR
    Initial gain (Min possible gain) is 15dB because LNA1 is ALWAYS kept ON. Thus,
    we have ATTN + LNA1 + LN2 + LNA + TIA + PGA:
        15 = -18 +   33 +   0 +   0 +   0 +  0 (setting of PGA=-12 is considered gain=0, it's not an attenuator)

    INPUT:
        - gain: Value between 0 and 108

    OUTPUT:
        - gain_setting: dict with all gain settings
    """
    gain = gain + 15
    filename = './gainTable_CBRS_08_26_19.csv'
    df = pd.read_csv(filename, header=None)

    # Get data
    # Format: total_gain, snr, LNA, TIA, PGA, LNA1, LNA2, ATTN
    total_gain = list(df.iloc[:, 0])
    idx = np.where(np.array(total_gain) == gain)[0]

    LNA = list(df.iloc[idx, 2])
    TIA = list(df.iloc[idx, 3])
    PGA = list(df.iloc[idx, 4])
    LNA1 = list(df.iloc[idx, 5])
    LNA2 = list(df.iloc[idx, 6])
    ATTN = list(df.iloc[idx, 7])
    gain_setting = dict({'LNA': LNA, 'TIA': TIA, 'PGA': PGA, 'LNA1': LNA1, 'LNA2': LNA2, 'ATTN': ATTN})

    return gain_setting


def set_lna_register(value):
    """
    ++++++++   LNA   ++++++++ 
    see: /usr/git/LimeSuite/src/lms7002m/LMS7002M.cpp

    Convert LNA setting to value directly written into fpga register
    Input: value between 0:1:30
    """
    gmax = 30
    val = value - gmax

    g_lna_rfe = 0
    if val >= 0:
        g_lna_rfe = 15
    elif val >= -1:
        g_lna_rfe = 14
    elif val >= -2:
        g_lna_rfe = 13
    elif val >= -3:
        g_lna_rfe = 12
    elif val >= -4:
        g_lna_rfe = 11
    elif val >= -5:
        g_lna_rfe = 10
    elif val >= -6:
        g_lna_rfe = 9
    elif val >= -9:
        g_lna_rfe = 8
    elif val >= -12:
        g_lna_rfe = 7
    elif val >= -15:
        g_lna_rfe = 6
    elif val >= -18:
        g_lna_rfe = 5
    elif val >= -21:
        g_lna_rfe = 4
    elif val >= -24:
        g_lna_rfe = 3
    elif val >= -27:
        g_lna_rfe = 2
    else:
        g_lna_rfe = 1

    valMod = g_lna_rfe
    return valMod


def set_tia_register(value):
    """
    ++++++++   TIA   ++++++++ 
    see: /usr/git/LimeSuite/src/lms7002m/LMS7002M.cpp

    Convert TIA setting to value directly written into fpga register
    Input: value in [0, 9, 12]
    """
    gmax = 12
    val = value - gmax

    g_tia_rfe = 0
    if val >= 0:
        g_tia_rfe = 3
    elif val >= -3:
        g_tia_rfe = 2
    else:
        g_tia_rfe = 1

    valMod = g_tia_rfe
    return valMod


def set_pga_register(value):
    """
    ++++++++   PGA   ++++++++ 
    see: /usr/git/LimeSuite/src/lms7002m/LMS7002M.cpp

    LMS7_RCC_CTL_PGA_RBB = { 0x011A, 13, 9, 23, "RCC_CTL_PGA_RBB", "Controls the stability passive compensation of the PGA_RBB operational amplifier" };
    LMS7_C_CTL_PGA_RBB = { 0x011A, 6, 0, 2, "C_CTL_PGA_RBB", "Control the value of the feedback capacitor of the PGA that is used to help against the 
    parasitic cap at the virtual node for stability" };
    float value[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    """
    arrayMod = []

    g_pga_rbb = (value + 12.5)
    if g_pga_rbb > 0x1f:
        g_pga_rbb = 0x1f
    if g_pga_rbb < 0:
        g_pga_rbb = 0

    rcc_ctl_pga_rbb = (430.0 * (0.65 ** (g_pga_rbb / 10.0)) - 110.35) / 20.4516 + 16

    c_ctl_pga_rbb = 0
    if 0 <= g_pga_rbb and g_pga_rbb < 8:
        c_ctl_pga_rbb = 3
    if 8 <= g_pga_rbb and g_pga_rbb < 13:
        c_ctl_pga_rbb = 2
    if 13 <= g_pga_rbb and g_pga_rbb < 21:
        c_ctl_pga_rbb = 1
    if 21 <= g_pga_rbb:
        c_ctl_pga_rbb = 0

    arrayMod.append(int(g_pga_rbb))
    arrayMod.append(int(rcc_ctl_pga_rbb))
    arrayMod.append(int(c_ctl_pga_rbb))
    return arrayMod


def set_lna1_register(gain_dB):
    """

        RF_MOD_CTRL2_LNA1_EN_SHIFT  = 0;
        RF_MOD_CTRL2_LNA1_EN_MASK   = 0x3;

    """
    reg = 0
    if gain_dB > 15.0:
        bit = 0x3
    else:
        bit = 0

    # SKLK_BITS(_ctrl2, bit, RF_MOD_CTRL2_LNA1_EN)
    bits = bit
    mask = 0x3
    shift = 0

    reg = ((reg & (~((mask) << (shift)))) | ((bits & (mask)) << (shift)))
    test = ((bits & (mask)) << (shift))
    print("LNA1 - REG: {} \t TEST: {}".format(reg, test))
    return bit, reg  # _ctrl2


def set_lna2_register(gain_dB):
    """
        RF_MOD_CTRL_LNA2_EN = (12)

    """
    reg = 0
    if gain_dB > 7.0:
        bit = 1
    else:
        bit = 0

    # SKLK_BIT(_ctrl0, bit, RF_MOD_CTRL_LNA2_EN)
    bits = bit
    mask = 0x1
    shift = 12

    reg = ((reg & (~((mask) << (shift)))) | ((bits & (mask)) << (shift)))
    test = ((bits & (mask)) << (shift))
    print("LNA2 - REG: {} \t TEST: {}".format(reg, test))
    return bit, reg  # _ctrl0


def set_attn_register(gain_dB):
    """
        RF_MOD_CTRL1_RXATTEN1_SHIFT = 10
        RF_MOD_CTRL1_RXATTEN1_MASK = (0x3)
        RF_MOD_CTRL1_RXATTEN2_SHIFT = 6
        RF_MOD_CTRL1_RXATTEN2_MASK = (0x3)

    """
    _revb = True   # CBRS rev B or C/E
    channel = 0
    reg = 0

    bits = round((18 + gain_dB) / 6.0)
    if bits > 3:
        bits = 3
    if bits < 0:
        bits = 0

    if not _revb:
        bits = (~bits) & 0x3  # revc part swap has inversion
    if channel == 0:
        # SKLK_BITS(_ctrl1, bits, RF_MOD_CTRL1_RXATTEN1)
        shift = 10
        mask = 0x3
    if channel == 1:
        # SKLK_BITS(_ctrl1, bits, RF_MOD_CTRL1_RXATTEN2)
        shift = 6
        mask = 0x3

    reg = ((reg & (~((mask) << (shift)))) | ((bits & (mask)) << (shift)))
    test = ((bits & (mask)) << (shift))
    print("ATTN - REG: {} \t TEST: {}".format(reg, test))
    return bits, reg  # _ctrl1


def get_lna_register(g_lna_rfe):
    """
        Convert LNA register value to actual gain
    """
    gmax = 30;
    
    if g_lna_rfe == 15:
        return gmax-0
    elif g_lna_rfe == 14:
        return gmax-1
    elif g_lna_rfe == 13: 
        return gmax-2
    elif g_lna_rfe == 12: 
        return gmax-3
    elif g_lna_rfe == 11: 
        return gmax-4
    elif g_lna_rfe == 10: 
        return gmax-5
    elif g_lna_rfe == 9: 
        return gmax-6
    elif g_lna_rfe == 8: 
        return gmax-9
    elif g_lna_rfe == 7: 
        return gmax-12
    elif g_lna_rfe == 6: 
        return gmax-15
    elif g_lna_rfe == 5: 
        return gmax-18
    elif g_lna_rfe == 4: 
        return gmax-21
    elif g_lna_rfe == 3: 
        return gmax-24
    elif g_lna_rfe == 2: 
        return gmax-27
    elif g_lna_rfe == 1: 
        return gmax-30
    else:
        return 0.0


def get_tia_register(g_tia_rfe):
    """
        Convert TIA register value to actual gain
    """
    gmax = 12;
    if g_tia_rfe == 3:
        return gmax-0
    elif g_tia_rfe == 2: 
        return gmax-3
    elif g_tia_rfe == 1: 
        return gmax-12
    else:
        return 0.0


def get_pga_register(g_pga_rbb):
    """
        Convert PGA register value to actual gain
    """
    return g_pga_rbb - 12


if __name__ == '__main__':
    # Test
    gain = 30
    gain_setting = gain_to_gain_setting(gain)

    # LNA
    lnavec = list(range(0, 31))
    for idx, val in enumerate(lnavec):
        out_lna = set_lna_register(val)
        print("LNA - IN:{} OUT:{}".format(val, out_lna))

    # TIA
    tiavec = [0, 9, 12]
    for idx, val in enumerate(tiavec):
        out_tia = set_tia_register(val)
        print("TIA - IN:{} OUT:{}".format(val, out_tia))

    # PGA
    pgavec = list(range(-12, 20))
    for idx, val in enumerate(pgavec):
        out_pga = set_pga_register(val)
        print("PGA - IN:{} OUT:{}".format(val, out_pga))

    # ATTN
    attnvec = [-18, -12, -6, 0]
    for idx, val in enumerate(attnvec):
        bit, out_attn = set_attn_register(val)
        #print("ATTN - IN:{} OUT:{}".format(val, out_attn))

    # LNA1
    lna1vec = [0, 33]
    for idx, val in enumerate(lna1vec):
        bit, out_lna1 = set_lna1_register(val)
        #print("LNA1 - IN:{} OUT:{}".format(val, out_lna1))

    # LNA2
    lna2vec = [0, 17]
    for idx, val in enumerate(lna2vec):
        bit, out_lna2 = set_lna2_register(val)
        #print("LNA2 - IN:{} OUT:{}".format(val, out_lna2))
