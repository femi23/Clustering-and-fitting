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

# to remove all non-countries from the dataset, store all non_countries into a list
not_country = [
    'Africa Eastern and Southern', 'World', 'Upper middle income', 
    'Sub-Saharan Africa (excluding high income)', 'Sub-Saharan Africa (IDA & IBRD countries)',
    'Sub-Saharan Africa', 'South Asia (IDA & IBRD)', 'South Asia', 'Small states', 
    'Post-demographic dividend', 'Pre-demographic dividend', 'Pacific island small states', 
    'Other small states', 'OECD members', 'Middle East & North Africa', 
    'Middle East & North Africa (IDA & IBRD countries)', 
    'Middle East & North Africa (excluding high income)', 'Middle income', 
    'Low & middle income', 'Low income', 'Lower middle income', 'Late-demographic dividend', 
    'Latin America & Caribbean', 'Latin America & Caribbean (excluding high income)', 
    'Latin America & the Caribbean (IDA & IBRD countries)', 'Least developed countries: UN classification', 
    'IBRD only', 'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total', 'Heavily indebted poor countries (HIPC)', 
    'Fragile and conflict affected situations', 'Euro area', 'Europe & Central Asia',
    'Europe & Central Asia (IDA & IBRD countries)', 'Europe & Central Asia (excluding high income)', 'European Union',
    'Early-demographic dividend', 'East Asia & Pacific', 'East Asia & Pacific (IDA & IBRD countries)',
    'East Asia & Pacific (excluding high income)', 'Caribbean small states', 'Central Europe and the Baltics', 
    'Africa Western and Central' 
    ]


def clean_df(df, indicators):
    '''
        Takes a dataframe and year and transforms it into a format ready for clustering which
        contains only data for the selected year
        
        Args:
            df => pandas.Dataframe, dataframe to be cleaned
            year => int, year to filter
        Returns:
            filtered_df => pandas.Dataframe, transformed dataframe
    '''
    # select only columns of interest, drop rows with missing values and non-countries
    filtered_df = df[['Country Name'] + indicators]
    filtered_df = filtered_df.dropna()
    filtered_df = filtered_df[~filtered_df['Country Name'].isin(not_country)]
        
    return filtered_df


# let's also select the variables/indicators we are interested in
int_ind = [
    'GDP per capita (current US$)', 'CO2 emissions (kg per PPP $ of GDP)', 
    'CO2 emissions (kt)', 'Population, total', 'Renewable energy consumption (% of total final energy consumption)'
    ]


clu_df = clean_df(tran_df, int_ind)
# Lets take a look at the tranformed dataframe and observe all countries in our dataset
print(clu_df.head())
print(clu_df.shape)


#adding another variable..
clu_df['CO2 production per head'] = clu_df['CO2 emissions (kt)']/clu_df['Population, total']


#lets check range for data to understand if normalization is needed
print(clu_df.describe().loc[['min', 'max'], :])


# let visualization 2 variables on scatter plot
def plot_scatterplot(df, x, y, color=None):
    '''
        Takes a dataframe and plot two of its variables on a scatter plot
        
        Args:
            df => pandas.Dataframe, dataframe to visualize
            x => first indicator
            y => second indicator
            color => color of point on plot
        Returns:
             grapgh => scatterplot
    '''
    plt.scatter(df[x],df[y], color = color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


# visualize data set on scatter plot before clustering
plot_scatterplot(clu_df, 'CO2 production per head', 'GDP per capita (current US$)')
plot_scatterplot(clu_df, 'Renewable energy consumption (% of total final energy consumption)', 'GDP per capita (current US$)')
plot_scatterplot(clu_df, 'CO2 emissions (kg per PPP $ of GDP)', 'GDP per capita (current US$)')


def cluster_dataframe(df, k, indicators):
    '''
        Takes a dataframe seperates it into k clusters
        
        Args:
            df => pandas.Dataframe, dataframe to be clustered
            k => int, number of clusters
            indicators => list, variables for clustering
        Returns:
             new_df => pandas.Dataframe, clustered dataframe
             centroid_df => pandas.Dataframe, centroids dataframe
    '''
    # cluster dataset using kmeans and save result to cluster column.
    newdf = df.copy()
    km = KMeans(n_clusters=k)
    cluster_group = km.fit_predict(newdf[indicators])
    newdf['cluster'] = cluster_group
    
    # save centroid into dataframe
    centroid_df = pd.DataFrame(km.cluster_centers_, columns = indicators)
    
    return newdf, centroid_df


var_ind = [
    'GDP per capita (current US$)', 'CO2 emissions (kg per PPP $ of GDP)', 
    'CO2 production per head', 'Renewable energy consumption (% of total final energy consumption)'
    ]


#cluster dataset into 3 seperate groups using the indicators above
clustered_df, centroid_df = cluster_dataframe(clu_df, 3, var_ind)
clustered_df.to_csv(f'Femi Cluster year {selected_year}.csv')


def visualize_clusters(df, centroid_df, k, colors, x, y, year):
    '''
        plots the clusters with different colors on a scattered plot 
        
        Args:
            df => pandas.Dataframe, clustered dataframe to be visualize
            centroid_df => pandas.Dataframe, dataframe to be transformed
            k => int, number of clusters
            x => first indicator
            y => second indicator
        Returns:
             grapgh => scatterplot
    '''
    for i in range(k):
        cluster = df[df.cluster==i]
        plt.scatter(cluster[x],cluster[y], color = colors[i], label = f'cluster {color_map[colors[i]]}')


    plt.scatter(centroid_df[x],centroid_df[y],color='purple',marker='*',label='centroid')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.title('Country clustering for the Year ' + str(year))
    plt.show()
    

plot_colors = ['#FAC012', '#2F528F', 'black']

color_map = {
    '#FAC012': 0, '#2F528F' : 1, 'black':2
    }

# visualize dataset on scatter plot after clustering
visualize_clusters(clustered_df, centroid_df, 3, plot_colors, 'CO2 production per head',  'GDP per capita (current US$)', selected_year)
visualize_clusters(clustered_df, centroid_df, 3, plot_colors, 'Renewable energy consumption (% of total final energy consumption)', 'GDP per capita (current US$)', selected_year)
visualize_clusters(clustered_df, centroid_df, 3, plot_colors, 'CO2 emissions (kg per PPP $ of GDP)', 'GDP per capita (current US$)', selected_year)


def get_cluster_stat(df):
    '''
        this function gives the statistic of the clusters such as count, mean and median
        
        Args:
            df => pd.DataFrame, The Dataframe to get stat of.
    
        Returns:
            stat_df => pd.DataFrame, a dataframe containing the statistical summary of the different clusters.    
    
    '''
    stat_df = df.groupby('cluster').agg(['mean', 'count', 'median'])
    return stat_df


cluster_stat = get_cluster_stat(clustered_df)
print(cluster_stat)
cluster_stat.to_csv(f'Femi cluster_stat year {selected_year}.csv')


#let take a look at a few countries in each clusters
cluster_2 = clustered_df[clustered_df['cluster'] == 2]
print(cluster_2.head())

cluster_1 = clustered_df[clustered_df['cluster'] == 1]
print(cluster_1.head())

cluster_0 = clustered_df[clustered_df['cluster'] == 0]
print(cluster_0.head())
