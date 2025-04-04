# This file contains auxiliary methods that we will use in our computations

import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from datamanagement.longTermPower.utils import context_path_for_PFC
from tgp_research_quantitative_tools.optionpricing.MonteCarlo.DiffusionProcess import MultipleAssetTwoFactorProcess

def compute_dicount_factors(rates, start_simulation_date, end_simulation_date, start_contract_date, end_contract_date, payment_delay):
    """
    This method computes the discount factors for the simulation and payment parts
    """
    # Get the reference rate to compute discount factors  
    reference_date = rates['Date'].min()
    
    # Get the list of currencies
    currencies = rates.columns[1:]
    
    discount_factors_currencies = {}
    
    for ccy in currencies:
        
        # Select the corresponding rate
        full_extended_rates_with_discount = rates.copy()
        full_extended_rates_with_discount['Rate'] = rates[ccy] / 100

        # Compute maturity in years
        full_extended_rates_with_discount['T'] = (full_extended_rates_with_discount['Date'] - reference_date).dt.days / 365
        full_extended_rates_with_discount['Discount Factor'] = np.exp(-full_extended_rates_with_discount['Rate'] * full_extended_rates_with_discount['T'])
        full_extended_rates_with_discount = full_extended_rates_with_discount.set_index('Date')
        
        # Initialize the discount matrix for the simulation part
        daily_dates = pd.date_range(start=start_simulation_date, end=end_simulation_date, freq='D')
        monthly_dates = pd.date_range(start=start_contract_date, end=end_contract_date, freq='MS')

        # Create a DataFrame to hold the adjusted dates
        adjusted_dates = monthly_dates + pd.Timedelta(days=payment_delay)

        # Create a DataFrame to hold the discount factors for the adjusted dates
        adjusted_discount_factors = full_extended_rates_with_discount.loc[adjusted_dates, 'Discount Factor'].values

        # Create a DataFrame to hold the discount factors for the original dates
        original_discount_factors = full_extended_rates_with_discount.loc[daily_dates, 'Discount Factor'].values[:, None]

        # Calculate the discount matrix using vectorized operations
        discount_matrix_payment = adjusted_discount_factors / original_discount_factors

        # Cap the discount to 1 (it is higher than 1 for matured forwards)
        discount_matrix_payment = np.minimum(discount_matrix_payment, 1)

        # Assign the calculated values back to the original DataFrame
        discount_matrix_payment = pd.DataFrame(discount_matrix_payment, index=daily_dates, columns=monthly_dates)
        
        # Save the discount factor for this currency
        discount_factors_currencies[ccy] = discount_matrix_payment
        
    return discount_factors_currencies

def compute_instantaneous_pd_from_cumulative(cumulative_pd_list):
    """
    Method to compute the instantaneous probability of default from the cumulative one
    """
    # Initialize the list of instantaneous PD with the first value of the cumulative PD
    instantaneous_pd = np.array(cumulative_pd_list)
    
    # Compute the instantaneous PD (this can be vectorized, but it is not too consuming and this is clear)
    for i in range(1, len(cumulative_pd_list)):
        instantaneous_pd[i] = 1 - (1 - instantaneous_pd[i]) / (1 - instantaneous_pd[:i]).prod()
    
    return instantaneous_pd

def compute_yearly_pd(pd_list, start_simulation_date):
    """
    Method to compute and align the yearly probability of default
    """
    # Compute the range of dates to assign the pd
    end_date = start_simulation_date + pd.DateOffset(years=len(pd_list))
    dates = pd.DataFrame(pd.date_range(start=start_simulation_date, end=end_date, freq='D'), columns=["index"])

    # We create the df of pd with one pd per 365 days
    rows = []
    for number in pd_list:
        rows.extend([number] * 365)
        
    pd_df = pd.DataFrame(rows, columns=['yearly_pd'])
    
    # Combine both dfs to assign each pd to one year starting on the first simulated date
    all_df = pd.concat([dates, pd_df], axis=1).dropna()

    return all_df

def compute_daily_pd(CVA_df, yearly_pd_df): 
    """
    Method to compute the daily probability of default from the yearly one
    """
    # Align the CVA df with the pd df and forward(backward)-fill the pds that we don't have
    combined_df = pd.merge(CVA_df, yearly_pd_df, on="index", how="left").set_index("index").ffill().bfill()
    
    # Compute the daily pd from the yearly pd
    daily_pd = 1 - (1 - combined_df["yearly_pd"]) ** (1 / 365)
    
    return daily_pd.values

def adjust_matured_trades(df, index_df, columns_df, payment_delay=0):
    """
    This method adjusts the matrix to account for matured trades by setting the values to 0 after the maturity date + payment_delay
    """
    if isinstance(df, np.ndarray):
        df_adjusted = df.copy()
    else:
        df_adjusted = df.values.copy()

    for col_index, col in enumerate(columns_df):
        try:
            # Get the row index where the column index matches
            match_index = index_df.get_loc(col)
            if match_index < len(df_adjusted):
                # Assign 0 to all values in the column after the matching index
                if df_adjusted.ndim == 3:
                    df_adjusted[match_index+payment_delay+1:, col_index, :] = 0
                else:
                    df_adjusted[match_index+payment_delay+1:, col_index] = 0
        except KeyError:
            # Handle KeyError if the column name is not found in the index
            continue

    return df_adjusted

def simulate_fwd_prices( 
    countries,
    start_simulation_date,
    end_simulation_date,
    method,
    granularity,
    list_of_dates = None,
    nb_months = 1,
    theta = None,
    as_of = None,
    start_fwd_date = None,
    nb_simulation = 5000,
    rebase = True
):  
    # First, we need to split between US and non US markets
    us_markets = [country for country in countries if country in ['ERCOTNORTH', 'ERCOTHOUHUB', 'HH_gas']]
    non_us_markets = [country for country in countries if country not in ['ERCOTNORTH', 'ERCOTHOUHUB', 'HH_gas']]
    
    # Now, we simulate using the MultipleAsset simulation all of the markets
    simulations = {}
        
    if us_markets:
        with context_path_for_PFC(True):
            simulation = MultipleAssetTwoFactorProcess.calibrate_and_simulate(
                us_markets,
                start_simulation_date,
                end_simulation_date,
                method,
                granularity,
                list_of_dates = list_of_dates,
                nb_months = nb_months,
                theta = theta,
                as_of = as_of - dt.timedelta(days=1), ## no curve yesterday in the US but only two days ago, weekend no supported here with this. TO FIX
                start_fwd_date = start_fwd_date,
                nb_simulation = nb_simulation,
                start_calibration_date = dt.date.today() + relativedelta(years=-2),
                end_calibration_date = dt.date.today() + relativedelta(days=-1),
                rebase=rebase,
                nb_months_calibration=20 # this part is sometimes needed and sometimes not, look at what is happening. TO FIX
            )
            
            simulations = {**simulation}

    if non_us_markets:
        # We first check that the countries are valid
        for country in non_us_markets:
            if country not in ['FR', 'DE', 'BE', 'UK', 'ES', 'TTF_gas', 'BRENT', 'JKM_gas']:
                raise ValueError("countries not supported, please choose between 'FR', 'DE', 'BE', 'UK', 'ES', 'TTF_gas', 'BRENT', 'JKM_gas'.")
        
        simulation = MultipleAssetTwoFactorProcess.calibrate_and_simulate(
            non_us_markets, 
            start_simulation_date,
            end_simulation_date, 
            method,
            granularity, 
            list_of_dates = list_of_dates,
            nb_months = nb_months,
            theta = theta,
            as_of=as_of, 
            start_fwd_date=start_fwd_date,
            nb_simulation=nb_simulation,
            start_calibration_date = dt.datetime.today() + relativedelta(years=-2),
            end_calibration_date = dt.datetime.today() + relativedelta(days=-1),
            rebase=rebase
        )
        
        simulations = {**simulations, **simulation}
        
    # Standarize the key names
    simulations = {k.split("-")[0]:v for k,v in simulations.items()}
        
    return simulations
