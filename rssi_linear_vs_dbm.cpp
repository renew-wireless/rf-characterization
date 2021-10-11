
/*
 *  rssi_linear_vs_dbm.cpp: Generate vector needed to convert from a given 
 *                          fixed point RSSI value to a gain value and viceversa. 
 *                          Table used in agc_core.sv to populate "rssi_array"
 *
 */


#include <stdio.h>
#include <cmath>
#include <iostream>
#include <math.h>
//#include "all_gain_tables.hpp"
#include <bitset>



int main()
{

    /* Convert from digital rssi value to Power in dBm (Matlab code below)
	rssi_fpga   = 5030;
	Vrms_fpga   = (rssi_fpga / 2.0^16) * (1 / sqrt(2)); %# Vrms = Vpeak/sqrt(2) (In Volts)
	PWRrms_fpga = (Vrms_fpga ^ 2.0) / 50.0;             %# 50 Ohms load (PWRrms in Watts)
	PWRdBm_fpga = 10.0 * log10(PWRrms_fpga) + 30;       %# P(dBm)=10*log10(Prms/1mW)   OR   P(dBm)=10*log10(Prms)+30
     */
    int rssi_fpga = 20724;
    double Vrms_fpga = (rssi_fpga / pow(2.0, 16.0)) * (1 / sqrt(2));
    double PWRrms_fpga = (pow(Vrms_fpga, 2.0)) / 50;
    double PWRdBm_fpga = 10 * log10(PWRrms_fpga) + 30;


    double PWRrms_fpga_rev = pow(10, (PWRdBm_fpga - 30)/10);
    double Vrms_fpga_rev = sqrt(50*PWRrms_fpga_rev);
    int rssi_fpga_rev = (Vrms_fpga_rev * sqrt(2)) * pow(2.0, 16.0);
    std::cout << "TEST - RSSIfp: "<< rssi_fpga << " PWRdBm_fpga: " << PWRdBm_fpga << " RSSIfp2:" << rssi_fpga_rev << std::endl;

    /* Do the opposite: Convert from dB value to digital rssi) */
    // dBm
    //int maxGain = 64;
    int maxGain = 108;
    int* rssi_fpga_arr_pt = new int[maxGain+1];
    std::cout << "XXX dBm XXX" << std::endl;
    for(int i=0; i>=-maxGain; i--)
    {
	    int i_abs = std::abs(i);
	    PWRdBm_fpga = (double) i;
	    PWRrms_fpga_rev = pow(10, (PWRdBm_fpga - 30)/10);
	    Vrms_fpga_rev = sqrt(50*PWRrms_fpga_rev);
	    rssi_fpga_arr_pt[i_abs] = (Vrms_fpga_rev * sqrt(2)) * pow(2.0, 16.0);
	
        // Convert to binary and create table hardcoded into FPGA
        std::bitset<8> aa(i_abs);
	    std::bitset<32> bb(rssi_fpga_arr_pt[i_abs]);

	    //std::cout << "32'b" << bb << ": gain_difference <= 8'b" << aa << ";  //Gain difference of (dB)" << i_abs << std::endl;
	    //std::cout << i_abs << " " << rssi_fpga_arr_pt[i_abs] << std::endl; 
	    std::cout << rssi_fpga_arr_pt[i_abs] << ", ";
    }
    std::cout << std::endl;

    // dBFS
    int maxGain2 = 108;
    int maxRssi = 21700;  // 20724 ;  // Highest values seen at around 21700 
    int* rssi_fpga_arr_pt2 = new int[maxGain2+1];
    double PWRdbFS;
    std::cout << "XXX dBFS XXX" << std::endl;
    for(int i=0; i>=-maxGain2; i--)
    {
        int i_abs = std::abs(i);
        PWRdbFS = (double) i;
	// std::cout << "i: " << i << " PWRdbFS: " << PWRdbFS << std::endl;
        // Formula: PWRdbFS = 20*log10(rssi/maxRssi)
        rssi_fpga_arr_pt2[i_abs] = pow(10, PWRdbFS / 20) * maxRssi;
        std::cout << rssi_fpga_arr_pt2[i_abs] << ", ";
    }
    std::cout << std::endl;

    return 0;
}
