
/*
 *  ctables_to_fpga.cpp: Performs several conversions needed for gainTable implemented on FPGA
 *
 *  1) Shows how to convert input taken by setGain() functions (used by firmware code as well 
 *     as other SoapyIris code) to actual value written to the LNA/TIA/PGA registers.
 *     This is essentially the conversion LimeSuite uses to go from one domain to the other. 
 *  2) Takes the values hardcoded into the gainTable and replaces them by their corresponding
 *     "register value" (value which will be written into register)
 *  3) Create gainTable that will be hardcoded into FPGA (converst decimal values to binary)
 * 
 */


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <bitset>
#include <vector>
#include <algorithm>
#include "gain_tables_all_NEW.hpp"    // CONTAINS OTHER DEFINITIONS
#include "gain_tables_oct_2021.hpp"


void verifyGainTables()
{
    int rowsRX_LO = sizeof(gainTableRx_LO) /sizeof(gainTableRx_LO[0]);
    int colsRX_LO = sizeof(gainTableRx_LO[0]) /sizeof(double);
    std::cout << "RX LO Size - Rows: " << rowsRX_LO << " Cols: " << colsRX_LO << std::endl;

    int rowsRX_HI = sizeof(gainTableRx_HI) /sizeof(gainTableRx_HI[0]);
    int colsRX_HI = sizeof(gainTableRx_HI[0]) /sizeof(double);
    std::cout << "RX HI Size - Rows: " << rowsRX_HI << " Cols: " << colsRX_HI << std::endl;

    int rowsTX_LO = sizeof(gainTableTx_LO) /sizeof(gainTableTx_LO[0]);
    int colsTX_LO = sizeof(gainTableTx_LO[0]) /sizeof(double);
    std::cout << "TX LO Size - Rows: " << rowsTX_LO << " Cols: " << colsTX_LO << std::endl;

    int rowsTX_HI = sizeof(gainTableTx_HI) /sizeof(gainTableTx_HI[0]);
    int colsTX_HI = sizeof(gainTableTx_HI[0]) /sizeof(double);
    std::cout << "TX HI Size - Rows: " << rowsTX_HI << " Cols: " << colsTX_HI << std::endl;

    std::cout << " ==========  RX LO  ==========" << std::endl;
    for (int i = 0; i < rowsRX_LO; i++){
        int sumval = 0;
        for (int j = 1; j < colsRX_LO; j++){
            sumval += gainTableRx_LO[i][j];
            //std::cout << "MEGD: " << gainTableRx_LO[i][j] << std::endl;
        }
        std::cout << "VAL: " << gainTableRx_LO[i][0] << " SUM: " << sumval << " EQUAL? " << (gainTableRx_LO[i][0]==sumval? 1:0) << std::endl;
    }

    std::cout << " ==========  RX HI  ==========" << std::endl;
    for (int i = 0; i < rowsRX_HI; i++){
        int sumval = 0;
        for (int j = 1; j < colsRX_HI; j++){
            sumval += gainTableRx_HI[i][j];
            //std::cout << "MEGD: " << gainTableRx_HI[i][j] << std::endl;
        }
        std::cout << "VAL: " << gainTableRx_HI[i][0] << " SUM: " << sumval << " EQUAL? " << (gainTableRx_HI[i][0]==sumval? 1:0) << std::endl;
    }

    std::cout << " ==========  TX LO  ==========" << std::endl;
    for (int i = 0; i < rowsTX_LO; i++){
        int sumval = 0;
        for (int j = 1; j < colsTX_LO; j++){
            sumval += ceil(gainTableTx_LO[i][j]);
            //std::cout << "MEGD: " << gainTableRx_HI[i][j] << std::endl;
        }
        std::cout << "VAL: " << gainTableTx_LO[i][0] << " SUM: " << sumval << " EQUAL? " << (gainTableTx_LO[i][0]==sumval? 1:0) << std::endl;
    }

    std::cout << " ==========  TX HI  ==========" << std::endl;
    for (int i = 0; i < rowsTX_HI; i++){
        int sumval = 0;
        for (int j = 1; j < colsTX_HI; j++){
            sumval += ceil(gainTableTx_HI[i][j]);
            //std::cout << "MEGD: " << gainTableRx_HI[i][j] << std::endl;
        }
        std::cout << "VAL: " << gainTableTx_HI[i][0] << " SUM: " << sumval << " EQUAL? " << (gainTableTx_HI[i][0]==sumval? 1:0) << std::endl;
    }

    // WRITE TO FILE (TO BE READ BY PYTHON SCRIPT)
    std::ofstream toPython_LO;
    toPython_LO.open ("gain_table_TX_LO.csv");
    for (int i = 0; i < rowsTX_LO; i++){
        for (int j = 0; j < colsTX_LO; j++){
            toPython_LO << gainTableTx_LO[i][j];
            toPython_LO << ",";
        }
        toPython_LO << "\n";
    }
    toPython_LO.close();

    std::ofstream toPython;
    toPython.open ("gain_table_TX_HI.csv");
    for (int i = 0; i < rowsTX_HI; i++){
        for (int j = 0; j < colsTX_HI; j++){
            toPython << gainTableTx_HI[i][j];
            toPython << ",";
        }
        toPython << "\n";
    }
    toPython.close();
}


void gen_all_gain_settings(float lna[], unsigned int lna_len,
                         float tia[], unsigned int tia_len,
			 float pga[], unsigned int pga_len, 
			 float lna1[], unsigned int lna1_len, 
			 float lna2[], unsigned int lna2_len,
			 float attn[], unsigned int attn_len)
{
    // Generate all possible gain setting combinations.
    // PRINT OUT ENTIRE TABLE
    std::cout << "int gainTable[NUM_GAIN_LEVELS][7] = {" << std::endl;
    for(int attnIdx=0; attnIdx<=attn_len-1; attnIdx++){
        for(int lna1Idx=0; lna1Idx<=lna1_len-1; lna1Idx++){        
            for(int lna2Idx=0; lna2Idx<=lna2_len-1; lna2Idx++){
                for(int lnaIdx=0; lnaIdx<=lna_len-1; lnaIdx++){
                    for(int tiaIdx=0; tiaIdx<=tia_len-1; tiaIdx++){
                        for(int pgaIdx=0; pgaIdx<=pga_len-1; pgaIdx++){
			    float sum_gain = attn[attnIdx] + lna1[lna1Idx] + lna2[lna2Idx] + lna[lnaIdx] + tia[tiaIdx] + pga[pgaIdx];
                            // For AGC purposes it is better if we start the "gain" value (first column) at zero, to do so, we need to account for the weird negative PGA setting, the LNA1 value, and the negative ATTN value
                            std::cout <<"{ \t "<< sum_gain-lna1[lna1Idx]+12+18 << " \t , \t " << attn[attnIdx] << " \t , \t " << lna1[lna1Idx] << " \t , \t " << lna2[lna2Idx] << " \t , \t " << lna[lnaIdx] << " \t , \t " << tia[tiaIdx] << " \t , \t " << pga[pgaIdx] << " \t }," << std::endl;
                        }
                    }
                }
            }
        }
    }
    std::cout << "};" << std::endl;
}


float convertLNA(float value)
{
    // ++++++++   LNA   ++++++++ see: /usr/git/LimeSuite/src/lms7002m/LMS7002M.cpp
    double gmax = 30;
    double val = value - gmax;

    int g_lna_rfe = 0;
    if (val >= 0) g_lna_rfe = 15;
    else if (val >= -1) g_lna_rfe = 14;
    else if (val >= -2) g_lna_rfe = 13;
    else if (val >= -3) g_lna_rfe = 12;
    else if (val >= -4) g_lna_rfe = 11;
    else if (val >= -5) g_lna_rfe = 10;
    else if (val >= -6) g_lna_rfe = 9;
    else if (val >= -9) g_lna_rfe = 8;
    else if (val >= -12) g_lna_rfe = 7;
    else if (val >= -15) g_lna_rfe = 6;
    else if (val >= -18) g_lna_rfe = 5;
    else if (val >= -21) g_lna_rfe = 4;
    else if (val >= -24) g_lna_rfe = 3;
    else if (val >= -27) g_lna_rfe = 2;
    else g_lna_rfe = 1;

    float valMod = g_lna_rfe;
    return valMod;
}

float convertTIA(float value)
{
    // ++++++++   TIA   ++++++++ see: /usr/git/LimeSuite/src/lms7002m/LMS7002M.cpp
    double gmax = 12;
    double val = value - gmax;

    int g_tia_rfe = 0;
    if (val >= 0) g_tia_rfe = 3;
    else if (val >= -3) g_tia_rfe = 2;
    else g_tia_rfe = 1;

    float valMod = g_tia_rfe;
    return valMod;
}

float* convertPGA(float value)
{
    // ++++++++   PGA   ++++++++
    // From /usr/git/LimeSuite/src/lime/LMS7002M_parameters.h:
    // LMS7_G_PGA_RBB = { 0x0119, 4, 0, 11, "G_PGA_RBB", "This is the gain of the PGA" };
    // LMS7_RCC_CTL_PGA_RBB = { 0x011A, 13, 9, 23, "RCC_CTL_PGA_RBB", "Controls the stability passive compensation of the PGA_RBB operational amplifier" };
    // LMS7_C_CTL_PGA_RBB = { 0x011A, 6, 0, 2, "C_CTL_PGA_RBB", "Control the value of the feedback capacitor of the PGA that is used to help against the 
    // parasitic cap at the virtual node for stability" };
    //float value[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    int size = 3; 	// 3 values needed
    float* arrayMod = new float[size];
    int g_pga_rbb = (int)(value + 12.5);
    if (g_pga_rbb > 0x1f) g_pga_rbb = 0x1f;
    if (g_pga_rbb < 0) g_pga_rbb = 0;

    int rcc_ctl_pga_rbb = (430.0*pow(0.65, (g_pga_rbb/10.0))-110.35)/20.4516 + 16;

    int c_ctl_pga_rbb = 0;
    if (0 <= g_pga_rbb && g_pga_rbb < 8) c_ctl_pga_rbb = 3;
    if (8 <= g_pga_rbb && g_pga_rbb < 13) c_ctl_pga_rbb = 2;
    if (13 <= g_pga_rbb && g_pga_rbb < 21) c_ctl_pga_rbb = 1;
    if (21 <= g_pga_rbb) c_ctl_pga_rbb = 0;

    arrayMod[0] = g_pga_rbb;
    arrayMod[1] = rcc_ctl_pga_rbb;
    arrayMod[2] = c_ctl_pga_rbb;
    return arrayMod;
}


// Code from /fw/iris030/sklk_fe/sklk_iris_fe_cbrs.cpp
int setCBRS_rxGain(int channel, std::string &name, float gain_dB)
{
    if (name == "ATTN")
    {
        //gain_dB = -18.0; //FIXME - REMOVE
        int bits = std::lround((18+gain_dB)/6.0);
        if (bits > 3) bits = 3;
        if (bits < 0) bits = 0;
        //if (not _revb) bits = (~bits) & 0x3; //revc part swap has inversion // DO THIS IN FPGA
        if (channel == 0) SKLK_BITS(_ctrl1, bits, RF_MOD_CTRL1_RXATTEN1);
        if (channel == 1) SKLK_BITS(_ctrl1, bits, RF_MOD_CTRL1_RXATTEN2);
	//std::cout << "ATTN: " << gain_dB << " CTRL1: " << _ctrl1 << std::endl;
        //_reg->wr(RF_MOD_CTRL1_ADDR, _ctrl1);
        return bits; //_ctrl1; 
    }
    else if (name == "LNA1")
    {
        int bit = (gain_dB > 15.0)?0x3:0; //TODO finer grained?
        SKLK_BITS(_ctrl2, bit, RF_MOD_CTRL2_LNA1_EN);
	//std::cout << "LNA1: " << gain_dB << " CTRL2: " << _ctrl2 << std::endl;
        //_reg->wr(RF_MOD_CTRL2_ADDR, _ctrl2);
        return bit; //_ctrl2;
    }
    else if (name == "LNA2")
    {
        //gain_dB = 0.0; //FIXME - REMOVE
        int bit = (gain_dB > 7.0)?1:0;
        SKLK_BIT(_ctrl0, bit, RF_MOD_CTRL_LNA2_EN);
	//std::cout << "LNA2: " << gain_dB << " CTRL0: " << _ctrl0 << std::endl;
        //_reg->wr(RF_MOD_CTRL0_ADDR, _ctrl0);
        return bit; //_ctrl0;
    }
    else throw std::runtime_error("SklkFrontEndCBRS::setGain("+name+") unknown");
}


void replaceRxGainTableVals(float freq)
{
    
    /* 
     * Get table to be translated from "gain_tables_all_NEW.hpp"
     * and translate
     *
     *  Table Formats: 
     * int gainTable[65][4]
     * int lnaTable[31][2]
     * int tiaTable[13][2]
     * int pgaTable[32][4]
     */

    float lna, tia, pga, attn, lna1, lna2; 
    int size;
    float lnaMod, tiaMod;
    float* pgaMod;
    int lna1Mod, lna2Mod, attnMod;
    std::string name;
    int channel = 0;

    auto gainTable = (freq == 2.5e9) ? gainTableRx_LO : gainTableRx_HI;
    int numGainLevels = (freq == 2.5e9) ? NUM_RX_GAIN_LEVELS_LO : NUM_RX_GAIN_LEVELS_HI;

    if(freq == 2.5e9){
        std::cout << "TABLE: LO" << std::endl;
    } else {
        std::cout << "TABLE: HI" << std::endl;
    }

    for(int idx=0; idx<=numGainLevels-1; idx++){
        // gainTable  format: [total_gain, ATTN, LNA1, LNA2, LNA, TIA, PGA]
        lna = gainTable[idx][4]; 
	    tia = gainTable[idx][5];
	    pga = gainTable[idx][6];
   	    lna1 = gainTable[idx][2];
	    lna2 = gainTable[idx][3];
	    attn = gainTable[idx][1];
	
	    lnaMod = convertLNA(lna);
	    tiaMod = convertTIA(tia);
	    pgaMod = convertPGA(pga);

        name = "LNA1";
        lna1Mod = setCBRS_rxGain(channel, name, lna1);
        name = "LNA2";
        lna2Mod = setCBRS_rxGain(channel, name, lna2);
        name = "ATTN";
        attnMod = setCBRS_rxGain(channel, name, attn);
	    
        //std::cout <<"LNA: "<<lna<<" -> "<<lnaMod<<" TIA: "<<tia<<" -> "<<tiaMod<<" PGA: "<<pga<<" -> "<<pgaMod[0]<<","<<pgaMod[1]<<","<<pgaMod[2] << " LNA1: " << lna1 << " -> " << lna1Mod << " LNA2: " << lna2 << " ->" << lna2Mod << " ATTN: "<< attn << " -> " << attnMod << std::endl;

        // Convert to binary and create table hardcoded into FPGA
        /*	    
        std::bitset<8> aa(idx);
        std::bitset<6> a(lnaMod);    // max setting == 15 => need 4 bits
	    std::bitset<6> b(tiaMod);    // max setting ==  3 => need 2 bits
	    std::bitset<5> c(pgaMod[0]); // max setting == 31 => need 5 bits
	    std::bitset<5> d(pgaMod[1]); // max setting == 31 => need 5 bits
	    std::bitset<2> e(pgaMod[2]); // max setting ==  3 => need 2 bits
	    std::bitset<2> f(lna1Mod);   // max setting ==  3 => need 2 bits
	    std::bitset<1> g(lna2Mod);   // max setting ==  1 => need 1 bits
	    std::bitset<2> h(attnMod);   // max setting ==  3 => need 2 bits
        */
	    std::bitset<8> aa(idx);
        std::bitset<5> a(lnaMod);    // max setting == 15 => need 4 bits
	    std::bitset<3> b(tiaMod);    // max setting ==  3 => need 2 bits
	    std::bitset<6> c(pgaMod[0]); // max setting == 31 => need 5 bits - g
	    std::bitset<6> d(pgaMod[1]); // max setting == 31 => need 5 bits - rcc
	    std::bitset<3> e(pgaMod[2]); // max setting ==  3 => need 2 bits - c
	    std::bitset<3> f(lna1Mod);   // max setting ==  3 => need 2 bits
	    std::bitset<2> g(lna2Mod);   // max setting ==  1 => need 1 bits
	    std::bitset<3> h(attnMod);   // max setting ==  3 => need 2 bits

	    ///// std::cout <<"LNA: "<<lna<<" -> "<< a <<" TIA: "<<tia<<" -> "<< b <<" PGA: "<<pga<<" -> "<< c <<","<< d <<","<< e <<std::endl; 
        std::cout << "8'b" << aa << ": return 32'b0" << f << g << h << a << b << c << d << e << ";  // Gain(dB)" << idx << std::endl; 
	    ///// std::cout << c << d << e << std::endl;
    }
    delete[] pgaMod;
}


void generate_tx_gain_table(float* attn, unsigned int attn_len,
                            float* pa1, unsigned int pa1_len,
                            float* pa2, unsigned int pa2_len,
                            float* pa3, unsigned int pa3_len,
                            float* pad, unsigned int pad_len,
                            float* iamp, unsigned int iamp_len)
{
    /* // DEPRECATED FUNCTION - NOW DONE IN TXRX_process_data.py script
     * Generate TX table for CBRS board. Main factor to consider is the noise figure of each component (each amplifier) in TX chain.
     * Currently, preference is as follows (from lower to higher noise).
     * Prioritize as follows (when possible):
     * PA2 -> ATTN -> PAD -> IAMP              (OLD & WRONG: IAMP > PAD > ATTN > PA2)
     */
    std::vector<double> totalgain_vec;
    int totalgain;

    std::cout << "const double gainTableTx_25G[NUM_TX_GAIN_LEVELS][7] = {" << std::endl;
    for(int pa1Idx=0; pa1Idx<=pa1_len-1; pa1Idx++){
        for(int pa3Idx=0; pa3Idx<=pa3_len-1; pa3Idx++){
            for(int pa2Idx=0; pa2Idx<=pa2_len-1; pa2Idx++){
                for(int attnIdx=0; attnIdx<=attn_len-1; attnIdx++){
                    for(int padIdx=0; padIdx<=pad_len-1; padIdx++){
                        for(int iampIdx=0; iampIdx<=iamp_len-1; iampIdx++){
                            // float sum_gain = attn[attnIdx] + lna1[lna1Idx] + lna2[lna2Idx] + lna[lnaIdx] + tia[tiaIdx] + pga[pgaIdx];
                            totalgain = (int) (pa1[pa1Idx] + pa3[pa3Idx] + pa2[pa2Idx] + attn[attnIdx] + pad[padIdx] + iamp[iampIdx]);
                            if (! std::count(totalgain_vec.begin(), totalgain_vec.end(), totalgain)) {                            
                                totalgain_vec.push_back(totalgain);
                                
                                std::cout <<"{ \t "<< totalgain << " \t , \t " << pa1[pa1Idx] << " \t , \t " << pa3[pa3Idx] << " \t , \t " << pa2[pa2Idx] << " \t , \t " << attn[attnIdx] << " \t , \t " << pad[padIdx] << " \t , \t " << iamp[iampIdx] << " \t }," << std::endl; 
                            }
                        }
                    }
                }
            }
        }
    }

    //for (std::vector<double>::const_iterator i = totalgain_vec.begin(); i != totalgain_vec.end(); ++i)
    //    std::cout << *i << '\n';


}


int main()
{

    /*
     * Specify gains and other settings
     */
    std::cout << "MAKE SURE TO SET CORRECT BOARD REV !!!" << std::endl;

    float freq = 3.6e9;  // Either 2.5GHz or 3.6GHz

    // RX
    float lna[] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27, 28, 29, 30};
    float tia[] = {0, 9, 12};
    float pga[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    
    float lna1[] = {33.0};
    float lna2[2];
    if(freq == 2.5e9){
        lna2[0] = 0.0;
        lna2[1] = 17.0;
   } else {
        lna2[0] = 0.0;
        lna2[1] = 14.0;
    }
    float attn[] = {-18.0, -12.0, -6.0, 0.0};
   
    // TX
    float iamp[] = {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float pad[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52};
    float pa1[1];
    float pa2[2];
    float pa3[1];
    if(freq == 2.5e9) {
        pa1[0] = {14};  // on/off
        pa2[0] = {0};   // bypass
        pa2[1] = {17};  // bypass
        pa3[0] = {31.5};// on/off
    } else {
        pa1[0] = {13.7};// on/off
        pa2[0] = {0};   // bypass
        pa2[1] = {14};  // bypass
        pa3[0] = {31};  // on/off
    }
    float attnTx[] = {-18.0, -12.0, -6.0};
    
    unsigned int lna_len = sizeof(lna)/sizeof(lna[0]);    // RX
    unsigned int tia_len = sizeof(lna)/sizeof(tia[0]);    // RX
    unsigned int pga_len = sizeof(lna)/sizeof(pga[0]);    // RX
    unsigned int lna1_len = sizeof(lna1)/sizeof(lna1[0]); // RX
    unsigned int lna2_len = sizeof(lna2)/sizeof(lna2[0]); // RX
    unsigned int attn_len = sizeof(attn)/sizeof(attn[0]); // RX
    unsigned int pad_len = sizeof(pad)/sizeof(pad[0]);    // TX
    unsigned int iamp_len = sizeof(iamp)/sizeof(iamp[0]); // TX
    unsigned int pa1_len = sizeof(pa1)/sizeof(pa1[0]);    // TX
    unsigned int pa2_len = sizeof(pa2)/sizeof(pa2[0]);    // TX
    unsigned int pa3_len = sizeof(pa3)/sizeof(pa3[0]);    // TX
    unsigned int attnTx_len = sizeof(attnTx)/sizeof(attnTx[0]); // TX

    /*
     *  Verify Gain Tables
     *
     */
    verifyGainTables();
    exit(0);

    /*
     * Specify gains and other settings
     */
    //gen_all_gain_settings(lna, lna_len, tia, tia_len, pga, pga_len, lna1, lna1_len, lna2, lna2_len, attn, attn_len);


    /*
     * (#1) - SHOW CONVERSION - Output not directly used to generate table
     * Show how to convert input taken by setGain() functions (used by firmware code as well 
     * as other SoapyIris code) to actual value written to the LNA/TIA/PGA registers.
     * This is essentially the conversion LimeSuite uses to go from one domain to the other.
     */

    // LMS7002M GAINS
    // LNA
    float lnaMod;
    int size = sizeof(lna)/sizeof(*lna)-1;
    for(int idx=0; idx<=size; idx++) {
    	lnaMod = convertLNA(lna[idx]);
        std::cout << "LNAsetting: " << lna[idx] << "\t| g_lna_rfe: " << lnaMod << std::endl;
        //std::cout << " { " << lna[idx] << " , " << lnaMod <<  " }, " << std::endl;  
    }

    // TIA
    float tiaMod;
    size = sizeof(tia)/sizeof(*tia)-1;
    for(int idx=0; idx<=size; idx++) {
    	tiaMod = convertTIA(tia[idx]);
        std::cout << "TIAsetting: " << tia[idx] << "\t| g_tia_rfe: " << tiaMod << std::endl;
        //std::cout << " { " <<value[idx] << " , " << g_tia_rfe <<  " }, " << std::endl;
    }

    // PGA
    float* pgaMod;
    size = sizeof(pga)/sizeof(*pga)-1;
    for(int idx=0; idx<=size; idx++) {
        pgaMod = convertPGA(pga[idx]);
        std::cout << "PGAsetting: " << pga[idx] << "\t| g_pga_rbb: " << pgaMod[0] << "\t| rcc_ctl_pga_rbb: " << pgaMod[1] << "\t| c_ctl_pga_rbb: " << pgaMod[2] << std::endl;
	std::bitset<5> c(pgaMod[0]);
        std::bitset<5> d(pgaMod[1]);
        std::bitset<2> e(pgaMod[2]);
        //std::cout << c << d << e << std::endl;
        //std::cout << " { " << value[idx] << " , " << g_pga_rbb << " , " << rcc_ctl_pga_rbb << " , " << c_ctl_pga_rbb << " }, " << std::endl;
    }

    // CBRS GAINS
    int channel = 0;
    std::string name;
   
    // ATTN
    int attnMod;
    name = "ATTN";
    size = sizeof(attn)/sizeof(*attn)-1;
    for(int idx=0; idx<=size; idx++) {
    	attnMod = setCBRS_rxGain(channel, name, attn[idx]);
        std::cout << "ATTNsetting: " << attn[idx] << "\t| g_attn_rfe: " << attnMod << std::endl;
    }

    // LNA1
    int lna1Mod;
    name = "LNA1";
    size = sizeof(lna1)/sizeof(*lna1)-1;
    for(int idx=0; idx<=size; idx++) {
    	lna1Mod = setCBRS_rxGain(channel, name, lna1[idx]);
        std::cout << "LNA1setting: " << lna1[idx] << "\t| g_lna1_rfe: " << lna1Mod << std::endl;
    }

    // LNA2
    int lna2Mod;
    name = "LNA2";
    size = sizeof(lna2)/sizeof(*lna2)-1;
    for(int idx=0; idx<=size; idx++) {
        lna2Mod = setCBRS_rxGain(channel, name, lna2[idx]);
        std::cout << "LNA2setting: " << lna2[idx] << "\t| g_lna2_rfe: " << lna2Mod << std::endl;
    }

    std::cout << " TABLE " << std::endl;
    /*
     * (#2) - ACTUAL TRANSLATION INTO FPGA TABLE
     * Takes the values hardcoded into the gainTable and replaces them by their corresponding
     * "register value" (value which will be written into register)
     * Create gainTable that will be hardcoded into FPGA (converst decimal values to binary)	
     */
    replaceRxGainTableVals(freq);

    /*
     * Generate TX gain table (DEPRECATED FUNCTION)
     */
    // generate_tx_gain_table(attnTx, attnTx_len, pa1, pa1_len, pa2, pa2_len, pa3, pa3_len, pad, pad_len, iamp, iamp_len);

    return 0;
}
