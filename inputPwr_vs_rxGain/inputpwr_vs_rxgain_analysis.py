#!/usr/bin/python
"""
 rxgain_vs_txpwr_analysis.py

    Analyze data collected

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
import seaborn as sns
from matplotlib.colors import ListedColormap
import pwlf

plt.style.use('fivethirtyeight')


def main():
    # Read data from CSV file
    data = pd.read_csv("./inputpwr_vs_rxgain_3.6GHz.csv")  # , encoding='utf8')
    pwr_at_ant = data.columns[0]
    rx_gains = data.columns[1::]

    # Create figure
    fig = plt.figure(1)
    ax = plt.axes()
    cmap = ListedColormap(sns.color_palette("hls", len(rx_gains)).as_hex())

    default = True
    if default:
        # WEIRD
        # Power at Antenna (Y-Axis), LMS7 Pwr (X-Axis)
        # Get data for each column
        data_all = np.transpose(np.array(data))
        data_y = data_all[0, :]  # data_all[1::, :]
        data_x = data_all[1::, :].astype(float)

        for idx, rx_gain in enumerate(rx_gains):
            y = data_y
            x = data_x[idx]

            # Initialize piecewise linear fit
            my_pwlf = pwlf.PiecewiseLinFit(x, y)
            # copy the x data to use as break points
            # breaks = my_pwlf.x_data.copy()
            # create the linear regression matrix A (not needed at the moment)
            # A = my_pwlf.assemble_regression_matrix(breaks, my_pwlf.x_data)

            per_segment = True
            if per_segment:
                # fit the data for 4 line segments
                num_segments = 4
                breaks = my_pwlf.fit(num_segments)
                beta = my_pwlf.beta
            else:
                # DEPRECATED
                # Break points:
                if int(rx_gain) == 0: x0 = np.array([min(x), -75.90, -72.01,  max(x)])
                elif int(rx_gain) == 10: x0 = np.array([min(x), -73.90, max(x)])
                # fit the data with the specified break points (ie the x locations of where
                # the line segments should end
                breaks = my_pwlf.fit_with_breaks(x0)

            # predict for the determined points
            x_hat = np.linspace(min(x), max(x), num=10000)
            y_hat = my_pwlf.predict(x_hat)

            # ==================
            # Check prediction of single point
            # check if x is numpy array, if not convert to numpy array
            x_verify = np.array([-70])
            if isinstance(x_verify, np.ndarray) is False:
                x_verify = np.array(x_verify)

            # B = my_pwlf.assemble_regression_matrix(breaks, x_verify)

            # Check if breaks in ndarray, if not convert to np.array
            if isinstance(breaks, np.ndarray) is False:
                breaks = np.array(breaks)
            # Sort the breaks, then store them
            breaks_order = np.argsort(breaks)
            fit_breaks = breaks[breaks_order]
            # store the number of parameters and line segments
            n_segments = len(breaks) - 1
            # Assemble the regression matrix
            A_list = [np.ones_like(x_verify)]
            A_list.append(x_verify - fit_breaks[0])
            for i in range(n_segments - 1):
                A_list.append(np.where(x_verify > fit_breaks[i + 1], x_verify - fit_breaks[i + 1], 0.0))
            A = np.vstack(A_list).T

            # solve the regression problem
            y_verify = np.dot(A, beta)
            # ==================

            print("==== STATS RxGain {} dB ====".format(rx_gain))
            print("Knee Points {}".format(breaks))
            print("Slopes {}".format(my_pwlf.slopes))
            print("Beta: {}".format(my_pwlf.beta))
            print("R-squared: {}".format(my_pwlf.r_squared()))

            # PLOTTER
            color = cmap(idx)
            print("Color: {}".format(color))
            plt.plot(x, y, color=color, linestyle=' ', marker='s', label=rx_gains[idx])
            plt.plot(x_hat, y_hat, color=color, linestyle='-')
            plt.plot(x_verify, y_verify, '*k')
        ax.set_ylabel("RX PWR (dBFS)")
        ax.set_xlabel("Input PWR at Antenna (dBm)")
        ax.legend(fontsize=10)
        ax.grid(True)
        plt.show()

    else:

        # Power at Antenna (X-Axis), LMS7 Pwr (Y-Axis)
        # Get data for each column
        data_all = np.transpose(np.array(data))
        data_y = data_all[1::, :]
        data_x = data_all[0, :].astype(float)

        for idx, rx_gain in enumerate(rx_gains):
            y = data_y[idx]

            # Initialize piecewise linear fit
            my_pwlf = pwlf.PiecewiseLinFit(data_x, y)

            # fit the data for 2 or 3 line segments
            # For low gain data two segments is enough, for high gain data we need three
            if int(rx_gain) <= 30:
                num_segments = 2
            else:
                num_segments = 4

            res = my_pwlf.fit(num_segments)

            # predict for the determined points
            x_hat = np.linspace(min(data_x), max(data_x), num=10000)
            y_hat = my_pwlf.predict(x_hat)

            # Get the slopes
            slopes = my_pwlf.slopes
            # Get my model parameters
            beta = my_pwlf.beta
            # calcualte the R^2 value
            Rsquared = my_pwlf.r_squared()
            print("==== STATS RxGain {} dB ====".format(rx_gain))
            print("Knee Points {}".format(res))
            print("Slopes {}".format(slopes))
            print("Beta: {}".format(beta))
            print("R-squared: {}".format(Rsquared))

            # PLOTTER
            color = cmap(idx / len(rx_gains))
            plt.plot(data_x, y, color=color, linestyle=' ', marker='s', label=rx_gains[idx])
            plt.plot(x_hat, y_hat, color=color, linestyle='-')
        ax.set_ylabel("RX PWR (dBFS)")
        ax.set_xlabel("Input PWR at Antenna (dBm)")
        ax.legend(fontsize=10)
        ax.grid(True)
        plt.show()


if __name__ == '__main__': 
    main()
