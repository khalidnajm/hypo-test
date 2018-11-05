import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Define Functions
# =============================================================================

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    #Number of data points
    n = len(data)
    #x-data for the ECDF
    x = np.sort(data)
    #y-data for the ECDF
    y = np.arange(1, n+1) / n

    return x, y

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1,perm_sample_2)

    return perm_replicates

def diff_of_means(data_1, data_2):
    """Calculate the difference in means from two data sets."""
    diff = np.mean(data_1) - np.mean(data_2)
    
    return diff

# =============================================================================
# Preparing data
# =============================================================================
    
#read in files and save as DataFrames
ob = pd.read_excel('OB_PREV_ALL_STATES.xlsx', header=1, index_col=[0,1], na_values='No Data')

#remove NAN values
ob = ob.dropna()

#prepare the DataFrame for analysis
#data of interest is data from 2013 only
ob = ob[['percent.9']]

#rename the column
ob.columns = ['Obesity (%)']
 
#EDA: plot ECDF of data
#rename Pandas Series for Ohio and Indiana for ease of use
ob_oh = ob.loc['Ohio','Obesity (%)']
ob_in = ob.loc['Indiana','Obesity (%)']

# =============================================================================
# EDA: Calulating Cumulative Distirbution Functions
# =============================================================================

#set the background
_ = sns.set()

#first, Ohio
x_oh, y_oh = ecdf(ob_oh)

#then, Indiana
x_in, y_in = ecdf(ob_in)

_ = plt.plot(x_oh, y_oh, marker = '.', linestyle = 'none', color = 'blue')
_ = plt.plot(x_in, y_in, marker = '.', linestyle = 'none', color = 'red')

#LABEL the axes
#_ = plt.xlabel('Percentage')
#_ = plt.ylabel('CDF')

#show the plot
plt.show()

# =============================================================================
# Analyis: Running a Permutation Test
# =============================================================================

#run permutation test 1000 times
perm_replicates = draw_perm_reps(np.array(ob_oh),
                                 np.array(ob_in),
                                                 diff_of_means, 
                                                 size = 1000)

#calculate p-value
p = np.sum(perm_replicates > diff_of_means(ob_oh, ob_in)) / len(perm_replicates)

print('p-value: ' + str(p))

