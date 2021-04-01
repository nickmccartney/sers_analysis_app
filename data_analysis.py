import pandas as pd
import numpy as np
from sklearn import pipeline

__all__ = [
    'resample_data',
    'append_to_dataset',
]

def resample_data(df, new_x = np.linspace(200, 1800, 800)):
    ###                                                                        ###
    #   returns df with each set of signal values interpolated over new_x points #
    #       * df : data to be resampled                                          #
    #       * new_x : optional specification of exact points to interpolate over #
    ###                                                                        ###

    old_x = df.columns    # gather values of raman shift sampled at
    old_y = df.values     # gather original signal values 

    new_y = [np.interp(new_x, old_x, signal) for signal in old_y]
    new_df = pd.DataFrame(columns=new_x, data=new_y)
    # df.columns = new_x    # modify sampling raman shift values to correspond to new_x values
    # df.data = new_y       # modifiy data to represent interpolated new_y values
    return new_df

def append_to_dataset(dataset, df, molecule, concentration, resample=False):
    df = df.transpose()      
    df.columns = df.iloc[0]                 # use raman shift row as column index
    print('HERE')
    print(df.values)
    df.drop('Raman Shift', inplace=True)    # FIXME: why is it now indexed by 'Raman' rather than 'Raman Shift'
    
    if resample and not dataset.empty:      # only allow resample if dataset has pre-existing raman shift values
        new_x = dataset.columns             # retrieve pre-existing raman shift values
        df = resample_data(df, new_x)       # interpolate df values in order to match to previous dataset entries

    if concentration in dataset.index.get_level_values(0).unique():                                 # check if concentration is already included in dataset
        n0 = int(dataset.loc[concentration].index[-1].replace('{} SP_'.format(molecule), '')) + 1   # find last value SP_n to start indexing sample at SP_n0 = n+1
    else:
        n0 = 0                                                                                      # otherwise, sample_idx starts at SP_0

    sample_idx = ['SP_{}'.format(n) for n in np.linspace(n0, n0+len(df.index), len(df.index), endpoint=False, dtype=int)]  # construct array of sample indicies
    molecule_idx = [molecule]*len(sample_idx)
    concentration_idx = [concentration]*len(sample_idx)                                             # construct array of concentration indices
    midx = pd.MultiIndex.from_arrays([molecule_idx, concentration_idx, sample_idx], names=('Molecule', 'Concentration', 'Sample'))      # combine concentration & sample in multi index
    df = df.set_index(midx)                 # assign multi index to df

    dataset = pd.concat([dataset, df])      # concatenate df onto dataset

    return dataset