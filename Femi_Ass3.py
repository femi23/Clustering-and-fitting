# Library for data processing
import numpy as np
import pandas as pd

# Libraries for data visualisation 
import matplotlib.pyplot as plt

# Libraries for data normalisation
from sklearn.preprocessing import MinMaxScaler


#Libraries for clustering and fitting
from sklearn.cluster import KMeans
import scipy.optimize as opt

# Libraries for error range
import err_ranges as err


def read_file(file_names):
    '''
        Takes a list of csv filenames, reads them into data frame and merges. 
        This is required when other indicators are in different files
        returns the merged dataframe and transposed format
        
        Args:
            filename => list, list of file names of csv files.
        Returns:
            final_df => pandas.Dataframe, merged dataframe
    '''
    
    dfs = []
    #read each csv files and return merged dataframe
    for csv_file in file_names:
        df = pd.read_csv(f'{csv_file}.csv')
        dfs.append(df)
    
    orig_df = pd.concat(dfs)
    # transposes the dataframe and modifies it
    transp_df = orig_df.set_index('Country Name').T
    transp_df.columns = [ transp_df.columns,  transp_df.iloc[1].tolist() ]
    transp_df.drop(['Country Code','Indicator Name', 'Indicator Code'], axis=0, inplace=True)
    transp_df = transp_df.apply(pd.to_numeric, errors='coerce')
    
    return orig_df, transp_df


# read two csv files, merge them and original and transposed
df, transp_df = read_file(['climate change dataset', 'GDP per capita'])
row, col = df.shape 

print(f'there are {row} rows and {col} columns in the climate change dataset')

## Lets take a look at the dataframe
print(df.head())


# Let's look at all the indicators
inidicator_count = df['Indicator Name'].nunique()
print(f'We have {inidicator_count} Indicators in this data set')
print(df['Indicator Name'].unique())


# Lets select a specific year to run the clustering on
selected_year = 2015


def transfrom_df(df, year):
    '''
        Takes a dataframe and year and transforms it into a format ready for clustering which
        contains only data for the selected year
        
        Args:
            df => pandas.Dataframe, dataframe to be transformed
            year => int, year to filter
        Returns:
            tran_df => pandas.Dataframe, transformed dataframe
    '''
    tran_df = df[['Country Name', 'Indicator Name', str(year)]]
    tran_df = tran_df.set_index(['Country Name', 'Indicator Name']).unstack()
    tran_df.columns = tran_df.columns.droplevel(0)
    tran_df = tran_df.reset_index()
    tran_df.columns.name = None
    
    return tran_df


tran_df = transfrom_df(df, selected_year)

# Lets take a look at the tranformed dataframe and observe all countries in our dataset
print(tran_df.head())
print(tran_df['Country Name'].unique())

