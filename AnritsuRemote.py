#!/usr/bin/python
"""

 tx_char_anritsu.py

    pip3 install PyVISA
    pip3 install PyVISA-py OR
    pip3 install -U https://github.com/hgrecco/pyvisa-py/zipball/master
    sudo pip3 install pyusb  (FOR REMOTE CONTROL VIA USB)

    COMMAND FOR TCPIP SOCKET
    TCPIP[board]::host address::port::SOCKET

    COMMAND FOR UBS INSTR:
    USB[board]::manufacturer ID::model code::serial number[::USB interface number][::INSTR]

    COMMAND FOR UBS RAW:
    USB[board]::manufacturer ID::model code::serial number[::USB interface number]::RAW



    Reference: MX269020A LTE FDD DL Remote Manual Document

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 ---------------------------------------------------------------------
"""

import sys
import pyvisa as visa
from pyvisa import util


class AnritsuRemote:

    def __init__(self, freq, bw, sig_type):
        print(" Initialize Device ")
        # Resource manager and logs
        util.get_debug_info()
        visa.log_to_screen()
        rm = visa.ResourceManager('@py')
        print("RESOURCES: {}".format(rm.list_resources()))
        # Raw socket set for Anritsu device: 49153
        # If Terminator is CR/LF, the terminator is a carriage return followed by a line feed.
        # device = rm.open_resource('TCPIP::169.254.183.116::49153::SOCKET', write_termination='\n', read_termination='\n')
        device = rm.open_resource('USB0::2907::6::6261932852::0::INSTR', write_termination='\n', read_termination='\n')
        device.write_termination = '\n'
        device.read_termination = '\n'
        # device.timeout = 5000

        try:
            id = str(device.query("*IDN?"))
        except:
            rm.close()
            print("QUERY FAILED")
            sys.exit()
        print("Found Device: {}".format(str(id)))

        # Remove carriage return (\r) and newline (\n)
        id = id.replace("\r\n", "").replace("\r", "").replace("\n", "")
        if id != "ANRITSU,MS2690A,6261932852,15.00.01":
            raise Exception("Anritsu device not found")

        # Set language mode (SCPI instead of native) and response mode
        device.write('INST CONFIG')
        device.write('SYST:LANG SCPI')
        device.write('SYST:RES:MODE A')

        # Start apps
        device.write('SYST:APPL:LOAD 3GLTE_DL')
        device.write('SYST:APPL:LOAD SIGANA')
        device.write('SYST:APPL:LOAD SPECT')

        self.device = device
        self.rm = rm
        self.freq = freq
        self.bw = bw
        self.sig_type = sig_type
        self.ampl = -140         # SigGen Output Power (dBm). Default to -140dBm
        self.f_err = []
        self.evm_vec = []

    def app_dl_lte3G(self):
        print(" ===== 3G LTE DL Test ===== ")
        # Select application and Initialize (Set Continuous or Single mode)
        self.device.write('INST 3GLTE_DL')
        app = self.device.query('INST?')
        app = app.replace("\r\n", "").replace("\r", "").replace("\n", "")
        print("Current APP: {} ".format(app))
        if app != "3GLTE_DL":
            raise Exception("Querying wrong application")

        self.device.write('*RST')
        self.device.write('*CLS')
        self.device.write('INIT:CONT ON')

        print(" Setting Basic Parameters ")
        self.set_basic_params()
        print(" Setting Common Modulation Parameters ")
        self.set_commom_modulation_params()

    def app_siggen_setup(self):
        print(" ===== SigGen ===== ")
        # Select application and Initialize
        self.device.write('INST SG')
        app = self.device.query('INST?')
        app = app.replace("\r\n", "").replace("\r", "").replace("\n", "")
        print("Current APP: {} ".format(app))
        if app != "SG":
            raise Exception("Querying wrong application")

        # Reset/Clear
        #self.device.write('*RST')
        #self.device.write('*CLS')

        # Set Frequency
        freq_str = 'FREQ ' + str(int(self.freq))
        self.device.write(freq_str)

        # Set Amplitude
        self.device.write('DISP:ANN:AMPL:UNIT DBM')
        ampl_str = 'POW ' + str(int(self.ampl)) + 'DBM'
        self.device.write(ampl_str)
        output_lvl = self.device.query('POW? DBM')
        print("Output Power Level: {} dBm".format(output_lvl))

        # Signal Type
        if self.sig_type == 'TM3_2':
            sig_str = 'E-TM_3-2_05M'
            sig_to_load = 'RAD:ARB:WAV "LTE_FDD","' + sig_str + '"'
        elif self.sig_type == 'WLAN':
            sig_str = '11a_OFDM_54Mbps'
            sig_to_load = 'RAD:ARB:WAV "WLAN","' + sig_str + '"'
        self.device.write(sig_to_load)

        # Enable modulation (no CW)
        self.device.write('OUTP:MOD ON')

    def siggen_output_onoff(self, new_state):
        # Signal Generator On/Off
        if new_state == 'ON':
            self.device.write('OUTP ON')
        else:
            self.device.write('OUTP OFF')
        sg_status = self.device.query('OUTP?')
        if sg_status == '1':
            sg_status = "ON"
        else:
            sg_status = "OFF"
        print("Signal Generator Turned {}".format(sg_status))

    def siggen_set_output_pwr(self):
        print(" ===== SigGen: Set Output Power ===== ")
        # Set Amplitude
        ampl_str = 'POW ' + str(int(self.ampl)) + 'DBM'
        self.device.write(ampl_str)
        output_lvl = self.device.query('POW? DBM')
        print("New Output Power Level: {} dBm".format(output_lvl))

    def set_basic_params(self):
        # Center Frequency
        freq_str = 'FREQ:CENT ' + str(int(self.freq))
        self.device.write(freq_str)
        # RF Spectrum (Normal NORM vs Reverse REV)
        self.device.write('SPEC NORM')
        # Input Level
        # self.device.write('POW:RANG:ILEV 0.00DBM')
        # Level Offset
        # self.device.write('DISP:WIND:TRAC:Y:RLEV:OFFS:STAT ON')
        # self.device.write('DISP:WIND:TRAC:Y:RLEV:OFFS 0.25DB')
        # Pre-Amp
        self.device.write('POW:GAIN OFF')

    def set_commom_modulation_params(self):
        # Trigger - Make sure it is OFF
        self.device.write('TRIG OFF')
        # device.write('TRIG:SOUR EXT')  # Trigger Source (External)
        # device.write('TRIG:SLOP POS')  # Trigger Slope
        # device.write('TRIG:DEL 0')     # Delay

        # Channel Bandwidth
        band_str = 'RAD:CBAN ' + str(int(self.bw))
        self.device.write(band_str)

        # Test Model (E-TM 3.2 16-QAM)
        sig_type_str = 'RAD:TMOD ' + self.sig_type
        self.device.write(sig_type_str)

        # PDSCH Modulation Scheme
        # self.device.write('CALC:EVM:PDSC:MOD AUTO')

        # Set Total EVM Calculation and EVM Window Length
        self.device.write('CALC:EVM:TEVM:RS INCL')
        self.device.write('CALC:EVM:TEVM:PDSC INCL')
        self.device.write('CALC:EVM:TEVM:PBCH INCL')
        self.device.write('CALC:EVM:TEVM:PSS INCL')
        self.device.write('CALC:EVM:TEVM:SSS INCL')
        self.device.write('CALC:EVM:TEVM:PDCC INCL')
        self.device.write('CALC:EVM:TEVM:PCF INCL')
        self.device.write('CALC:EVM:TEVM:PHIC INCL')
        self.device.write('CALC:EVM:WLEN 10')

        # Display EVM vs Resource Block
        self.device.write('DISP:EVM:SEL EVRB')

        # Setting Channel
        # All Available when Test Model set to Off
        # self.device.write('CALC:EVM:PBCH ON')
        # self.device.write('CALC:EVM:PBCH:POW:AUTO ON')
        # self.device.write('CALC:EVM:PSS ON')
        # self.device.write('CALC:EVM:PSS:POW:AUTO ON')
        # self.device.write('CALC:EVM:SSS ON')
        # self.device.write('CALC:EVM:SSS:POW:AUTO ON')
        # self.device.write('CALC:EVM:PHIC ON')
        # self.device.write('CALC:EVM:PHIC:POW:AUTO ON')
        # self.device.write('CALC:EVM:PHIC:NG R1BY6')
        # self.device.write('CALC:EVM:PDCC ON')
        # self.device.write('CALC:EVM:PDCC:POW:AUTO ON')
        # self.device.write('CALC:EVM:PDCC:SYMB AUTO')
        # self.device.write('CALC:EVM:PCF ON')
        # self.device.write('CALC:EVM:PCF:POW:AUTO ON')
        # self.device.write('CALC:EVM:PRS:STAN R8V830')  # Available when Test Model set to Off

        # Channel Estimation
        self.device.write('CALC:EVM:CHAN:EST ON')

    def modulation_measurements(self):
        """
        Results from device.query('READ:EVM?')
        1. Frequency Error (Average) [Hz]
        2. Frequency Error (Maximum) [Hz]
        3. Output Power (Average) [dBm]
        4. Output Power (Maximum) [dBm]
        5. Mean Power (Average) [dBm]
        6. Mean Power (Maximum) [dBm]
        7. EVM rms (Average) [%]
        8. EVM rms (Maximum) [%]
        9. EVM peak (Average) [%]
        10. EVM peak (Maximum) [%]
        11. EVM peak Symbol Number
        12. EVM peak Subcarrier Number
        13. Origin Offset (Average) [dB]
        14. Origin Offset (Maximum) [dB]
        15. Time Offset (Average) [seconds]
        16. Time Offset (Maximum) [seconds]
        17. Frequency Error PPM (Average) [ppm]
        18. Frequency Error PPM (Maximum) [ppm]
        19. Symbol Clock Error (Average) [ppm]
        20. Symbol Clock Error (Maximum) [ppm]
        21. IQ Skew (Average) [seconds]
        22. IQ Skew (Maximum) [seconds]
        23. IQ Imbalance (Average) [dB]
        24. IQ Imbalance (Maximum) [dB]
        25. IQ Quadrature Error (Average) [degree]
        26. IQ Quadrature Error (Maximum) [degree]
        """

        self.device.write('POWer:RANGe:AUTO ONCE')

        # Measurement function
        self.device.write('CONF:EVM')
        self.device.write('INIT:EVM')

        # Measurement Params
        self.device.write('EVM:CAPT:TIME:STAR 2')  # Sets analysis start time (subframe number, range from 0 to 9)
        self.device.write('EVM:CAPT:TIME:LENG 2')  # Analysis length (subframes)
        self.device.write('EVM:AVER ON')           # EVM Average ON
        self.device.write('EVM:AVER:COUN 10')      # Average count

        # Display
        # Trace Mode: EVM vs Resource Block
        self.device.write('DISP:EVM:SEL EVRB')

        # Scale–EVM Unit
        self.device.write('DISP:EVM:WIND2:TRAC:Y:SPAC PERCent')

        # Scale–EVM - reference level
        self.device.write('DISP:EVM:WIND2:TRAC:Y:RLEV 0')

        # Graph view setting
        self.device.write('CALC:EVM:WIND2:MODE AVERage')

        # Constellation Display Range
        self.device.write('DISP:EVM:WIND1:RANG SYMB')

        # Marker Symbol Number
        self.device.write('CALC:EVM:WIND2:SYMB:NUMB 1')

        # Marker On/Off
        self.device.write('CALC:EVM:MARK ON')
        self.device.write('CALC:EVM:MARK:ACT CONS')
        self.device.write('CALC:EVM:MARK:SYMB 1')

        # Read out measurements
        self.evm_vec = self.device.query('READ:EVM?')
        # self.f_err = self.device.query('STAT:ERR?')

        # Marker Value
        # rms_x = self.device.query('CALC:EVM:MARK:X?')
        # rms_y = self.device.query('CALC:EVM:MARK:Y?')
        # evm_marker = self.device.query('CALC:EVM:MARK:EVM?')
        # pwr = self.device.query('CALC:EVM:MARK:POW?')
        # chan = self.device.query('CALC:EVM:MARK:CHAN?')

        # print(" ---- Modulation Measurements ---- ")
        # print("EVM: {}".format(self.evm_vec))
        # print("Freq. Err: {}".format(self.f_err))
        # print("OTHER: {}, {}, {}, {}, {}".format(rms_x, rms_y, evm_marker, pwr, chan))

    def pwr_measurements(self):
        # Select application and measurement function
        self.device.write('CONF:FFT:CHP')

        # Set measurement parameters
        self.device.write('TRAC:STOR:MODE MAXH')
        self.device.write('AVER:COUN 10')

        # Read out measurements
        ch_pwr = self.device.query('READ:CHP?')
        stat_err = self.device.query('STAT:ERR?')

    def print_params(self):
        # Read back parameters
        fc = self.device.query('FREQ:CENT?')
        rf_spec = self.device.query('SPEC?')
        pwr_range_lvl = self.device.query('POW:RANG:ILEV?')
        ref_lvl = self.device.query('DISP:WIND:TRAC:Y:RLEV?')
        pre_amp = self.device.query('POW:GAIN?')
        pdsch_mod_check = self.device.query('CALC:EVM:PDSC:MOD?')
        fft_winw_len = self.device.query('CALC:EVM:WLEN:W?')

        # Print
        print(" ===== BASIC PARAMETERS ===== ")
        print("Center Freq: {}".format(fc))
        print("RF Spectrum: {}".format(rf_spec))
        print("PWR Input Level: {}".format(pwr_range_lvl))
        print("Reference Level: {}".format(ref_lvl))
        print("Pre-Amp State (0:OFF, 1:ON): {}".format(pre_amp))
        print("Modulation: {}".format(pdsch_mod_check))
        print("FFT Window W Len: {}".format(fft_winw_len))


#########################################
#                 Main                  #
#########################################
def main():

    """
        Basic Test Process:
        1) Initialization
        2) Select App
        3) Setting of Basic Parameters
        4) Setting of Modulation-Common Parameters
        5) Modulation Measurement
        6) Channel Power Measurement?
        7) Change Conditions and Go Back to Step 2

    """
    freq = '3.6 GHz'      # 2.5GHz or 3.6GHz
    bw = '5'              # in MHz
    sig_type = 'TM3_2'    # TM3_1, TM3_2, TM3_3
    # Anritsu Remote object
    ar = AnritsuRemote
    ar.app_dl_lte3G(freq, bw, sig_type)
    print("DONE")


if __name__ == '__main__':
    main()