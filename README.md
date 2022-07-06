# Python Project 2: Portfolio Optimization and Analysis

**Author**: Chinh X. Mai, **Date**: July 3, 2022

![image](https://user-images.githubusercontent.com/89245616/177422654-2da72da3-21b9-47ae-b553-82968ce75d7a.png)

## Project Description

This project focuses on constructing, optimizing, and analyzing a portfolio of chosen stocks in Python. Considering a realistic scenario when one has a certain amount of fund and wants to invest in the stock market either to make profit or to keep its value. He needs to know which stocks to buy and by how much. His friend has chosen some potential stocks (Table 1) and now he only needs to decide how to spend the money.


| Symbol  | Company Name                                | Last Price | Change | % Change | Volume     |
| ------- | ------------------------------------------- | ---------- | ------ | -------- | ---------- |
| ALV.DE  | Allianz SE                                  | 181.04     | \-0.16 | \-0.09%  | 1,069,053  |
| MRK.DE  | MERCK Kommanditgesellschaft auf Aktien      | 165.1      | 0.3    | 0.18%    | 313,644    |
| DTE.DE  | Deutsche Telekom AG                         | 18.84      | \-0.04 | \-0.20%  | 6,880,550  |
| VOW3.DE | Volkswagen AG                               | 138.88     | \-0.3  | \-0.22%  | 913,070    |
| DBK.DE  | Deutsche Bank Aktiengesellschaft            | 8.89       | \-0.03 | \-0.33%  | 10,112,593 |
| HNR1.DE | Hannover Rück SE                            | 136.4      | \-0.45 | \-0.33%  | 95,892     |

**Table 1**: list of suggested stocks to invest in (source: [Yahoo! Finance](https://finance.yahoo.com/quote/%5EGDAXI/components?p=%5EGDAXI), accessed on June 27, 2022)

These stocks are some components of the DAX30 Index, which includes many German blue chip companies trading on the Frankfurt Stock Exchange. This analysis will go through the process of constructing and validating the performance of a stocks portfolio that can serve the needs of the investor. This process will also provide many insights for the investor to understand the investment strategy that he can take to adjust the performance of the portfolio.

## Objectives

This project demonstrates my familiarity with many python packages used to perform a wide range of tasks necessary to optimize a portfolio. Besides the core packages used to calculate and manipulate data such as `pandas` and `numpy`, `plotly` is also used extensively to visualize stock data in the analysis and many simulations. Moreover, many utility functions and a class are generated from functions in `pypfopt` to simulate and present the optimization results. In details, the project aims to achieve the following objectives:

* Fetching data of chosen stocks from online source and save the data in a pandas DataFrame for further manipulations and calculations
* Investigating and cleaning the data for further analyses
* Understanding stock characteristics by calculations and visualizations
* Creating and optimizing stock portfolios for different risk preferences
* Validating and comparing the ex-ante performance of these portfolios
* Constructing an efficient frontier from the chosen stocks

Besides these objectives, the project also showcases my ability to utilize flexibly different tools provided by different packages in Python and present the results in an authentic and aesthetic manner.

## Detailed documentation

For detailed documentation, please refer to the project repository using the following link:

[Python Project 2 Github](https://github.com/ChinhMaiGit/Project-Python-2/)

or access the analysis workbook directly

[Python Project 2 Workbook](/html/project2.html)
