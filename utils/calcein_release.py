"""
Support for calcein release assays.
"""
import pandas as pd
import numpy as np
from scipy import stats
from os.path import Path
import glob

def calculate_calcein_release(
    data_dir: Path) -> pd.DataFrame:
    """
    Calculate the calcein release from the before, timecourse, and after dataframes.

    Args:
        data_dir: Path to the data directory.

    Returns:
        DataFrame containing the calcein release over time for each cell.
    """
    
    # Find the data files
    before_files = glob.glob(str(data_dir / "*before*.csv"))
    timecourse_files = glob.glob(str(data_dir / "*timecourse*.csv"))
    triton_files = glob.glob(str(data_dir / "*triton*.csv"))
    
    if not before_files or not timecourse_files or not triton_files:
        raise FileNotFoundError(f"Could not find required files in {data_dir}")
    
    # Use the first matching file for each type
    before_file = before_files[0]
    timecourse_file = timecourse_files[0]
    triton_file = triton_files[0]
    
    # Read the data files
    before_data = pd.read_csv(before_file, header=None)
    timecourse_data = pd.read_csv(timecourse_file, header=None)
    triton_data = pd.read_csv(triton_file, header=None)
    
    # Extract sample names from the first row
    sample_names = []
    for i in range(0, len(before_data.columns), 2):
        if i < len(before_data.columns) - 1:
            sample_name = before_data.iloc[0, i].strip()
            if sample_name.startswith('Sample '):
                sample_names.append(sample_name.replace('Sample ', ''))
    
    # Extract time and intensity data
    # Data starts from row 2 (index 1) and has time, intensity pairs for each sample
    def extract_fluorescence_data(data, sample_names):
        """Extract fluorescence data for each sample"""
        fluorescence_data = {}
        
        for i, sample in enumerate(sample_names):
            # Each sample has 2 columns: time and intensity
            time_col = 2 * i
            intensity_col = 2 * i + 1
            
            if time_col < len(data.columns) and intensity_col < len(data.columns):
                # Skip the header rows and get the actual data
                # Find where the data starts (after the metadata)
                data_start = 2  # Start after the header rows
                
                # Get time and intensity values
                times = data.iloc[data_start:, time_col].astype(float)
                intensities = data.iloc[data_start:, intensity_col].astype(float)
                
                # Remove any NaN values
                valid_mask = ~(times.isna() | intensities.isna())
                fluorescence_data[sample] = {
                    'time': times[valid_mask].values,
                    'intensity': intensities[valid_mask].values
                }
        
        return fluorescence_data
    
    # Extract data from all three files
    before_fluorescence = extract_fluorescence_data(before_data, sample_names)
    timecourse_fluorescence = extract_fluorescence_data(timecourse_data, sample_names)
    triton_fluorescence = extract_fluorescence_data(triton_data, sample_names)
    
    # Calculate average before and triton values for each sample
    before_avg = {}
    triton_avg = {}
    
    for sample in sample_names:
        if sample in before_fluorescence:
            before_avg[sample] = np.mean(before_fluorescence[sample]['intensity'])
        if sample in triton_fluorescence:
            triton_avg[sample] = np.mean(triton_fluorescence[sample]['intensity'])
    
    # Calculate calcein release percentage
    release_data = {}
    
    # Use time from timecourse data
    if sample_names and sample_names[0] in timecourse_fluorescence:
        release_data['time'] = timecourse_fluorescence[sample_names[0]]['time']
    
    for sample in sample_names:
        if (sample in timecourse_fluorescence and 
            sample in before_avg and 
            sample in triton_avg):
            
            f_timecourse = timecourse_fluorescence[sample]['intensity']
            f_before = before_avg[sample]
            f_triton = triton_avg[sample]
            
            # Calculate release percentage
            # release = 100 * (f_timecourse - f_before) / (f_triton - f_before)
            release_percentage = 100 * (f_timecourse - f_before) / (f_triton - f_before)
            
            release_data[sample] = release_percentage
    
    # Create the final DataFrame
    result_df = pd.DataFrame(release_data)
    
    return result_df


