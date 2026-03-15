"""
Data processing utilities for DNA brush paper analysis
"""

import pandas as pd
import numpy as np
from scipy import stats


def process_brush_data(
    df,
    dna_concentration_uM=5.0,
    lipid_volume_uL=5.0,
    lipid_concentration_mg_mL=0.5,
    lipid_molar_mass=760,
    control_name="ctrl",
):
    """
    Process brush DLS data by subtracting controls and calculating ratios

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data with columns including 'sample_type', 'sample_number',
        'batch', 'peak_1_mean_intensity'
    dna_concentration_uM : float
        DNA stock concentration in micromolar (default 5.0)
    lipid_volume_uL : float
        Volume of lipid solution in microliters (default 5.0)
    lipid_concentration_mg_mL : float
        Lipid concentration in mg/mL (default 0.5)
    lipid_molar_mass : float
        Lipid molar mass in g/mol (default 760 for POPC)
    control_name : str
        Name used for control samples (default "ctrl")

    Returns:
    --------
    pandas.DataFrame
        Processed data with controls subtracted and ratios calculated
    """
    controls = df["sample_type"] == control_name

    def process_row(row, controls_df):
        """Subtract control values from each row"""
        control_data = controls_df[controls_df["batch"] == row["batch"]]
        subtract_columns = [
            c not in ["batch", "sample_type", "sample_number"]
            for c in control_data.columns
        ]
        control_row = control_data.loc[:, subtract_columns].mean()

        new_row = row.copy()
        new_row[subtract_columns] = new_row[subtract_columns] - control_row
        return new_row

    df_ctrl = df[controls]
    df_data = df[~controls].apply(lambda row: process_row(row, df_ctrl), axis=1)

    # Calculate moles and ratios
    lipid_moles = lipid_volume_uL * 1e-6 * lipid_concentration_mg_mL / lipid_molar_mass

    # Convert sample_number to float, handling cases like "1.65.01"
    def clean_sample_number(x):
        x_str = str(x)
        try:
            return float(x_str)
        except ValueError:
            # Handle cases like "1.65.01" by removing extra decimals
            parts = x_str.split(".")
            if len(parts) > 2:
                return float(parts[0] + "." + parts[1])
            return float(x_str)

    df_data["sample_number_clean"] = df_data["sample_number"].apply(clean_sample_number)
    df_data["dna_moles"] = (
        df_data["sample_number_clean"] * 1e-6 * dna_concentration_uM * 1e-6
    )
    df_data["Lipid:Construct Ratio"] = (lipid_moles / df_data["dna_moles"]).astype(int)
    df_data["Concentration"] = (
        1 / df_data["Lipid:Construct Ratio"]
    )  # Concentration is inverse of ratio
    df_data["Delta D"] = df_data["peak_1_mean_intensity"]

    return df_data


def process_length_data(df, **kwargs):
    """
    Process brush length data (continuous categories)

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data
    **kwargs : dict
        Additional arguments for process_brush_data

    Returns:
    --------
    pandas.DataFrame
        Processed data with brush length as float
    """
    df_processed = process_brush_data(df, **kwargs)
    df_processed["Construct Length / bp"] = list(map(float, df_processed["sample_type"]))
    return df_processed


def process_shape_data(df, **kwargs):
    """
    Process brush shape data (categorical)

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data
    **kwargs : dict
        Additional arguments for process_brush_data

    Returns:
    --------
    pandas.DataFrame
        Processed data with shape as categorical
    """
    df_processed = process_brush_data(df, **kwargs)
    df_processed["Shape"] = df_processed["sample_type"]
    return df_processed


def process_charged_liposome_data(df):
    """
    Process charged liposome data with condition-specific controls

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data with columns: batch, condition, sample_type, z-average, p1mi

    Returns:
    --------
    pandas.DataFrame
        Processed data with controls subtracted and DOPG percentage calculated
    """
    # Clean data - handle potential BOM and formatting issues
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace("﻿", "")

    # Process each condition separately since controls are condition-specific
    processed_data = []

    for condition in df["condition"].unique():
        condition_data = df[df["condition"] == condition].copy()

        # Get controls for this condition
        controls = condition_data[condition_data["sample_type"] == "ctrl"]
        samples = condition_data[condition_data["sample_type"] != "ctrl"]

        # Subtract mean control values from samples in each batch
        for batch in samples["batch"].unique():
            batch_controls = controls[controls["batch"] == batch]
            batch_samples = samples[samples["batch"] == batch].copy()

            if len(batch_controls) > 0:
                # Calculate mean control values for this batch
                control_z_avg = batch_controls["z-average"].mean()
                control_p1mi = batch_controls["p1mi"].mean()

                # Subtract from samples
                batch_samples.loc[:, "delta_z_average"] = (
                    batch_samples["z-average"] - control_z_avg
                )
                batch_samples.loc[:, "delta_p1mi"] = (
                    batch_samples["p1mi"] - control_p1mi
                )

                processed_data.append(batch_samples)

    df_processed = pd.concat(processed_data, ignore_index=True)

    # Convert condition ratios to DOPG percentage
    # 1:0 = 0% DOPG, 9:1 = 10% DOPG, 3:1 = 25% DOPG, 1:1 = 50% DOPG
    condition_to_dopg = {"1:0": 0, "9:1": 10, "3:1": 25, "1:1": 50}
    df_processed["DOPG_percentage"] = df_processed["condition"].map(
        lambda x: condition_to_dopg.get(x)
    )

    return df_processed


def process_ph_data(df):
    """
    Process pH response data with condition-specific controls

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data with columns: batch, sample_type, condition, z-average, p1mi

    Returns:
    --------
    pandas.DataFrame
        Processed data with controls subtracted
    """
    # Clean data - handle potential BOM and formatting issues
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace("﻿", "")

    # Remove malformed rows (like the one with "25.01" in condition)
    df = df[df["condition"].astype(str).str.match(r"^\d+\.\d+$")].copy()
    df["condition"] = pd.to_numeric(df["condition"])

    # Process each pH condition separately since controls are condition-specific
    processed_data = []

    for condition in df["condition"].unique():
        condition_data = df[df["condition"] == condition].copy()

        # Get controls for this condition
        controls = condition_data[condition_data["sample_type"] == "ctrl"]
        samples = condition_data[condition_data["sample_type"] != "ctrl"]

        # Subtract mean control values from samples in each batch
        for batch in samples["batch"].unique():
            batch_controls = controls[controls["batch"] == batch]
            batch_samples = samples[samples["batch"] == batch].copy()

            if len(batch_controls) > 0:
                # Calculate mean control values for this batch
                control_z_avg = batch_controls["z-average"].mean()
                control_p1mi = batch_controls["p1mi"].mean()

                # Subtract from samples
                batch_samples.loc[:, "delta_z_average"] = (
                    batch_samples["z-average"] - control_z_avg
                )
                batch_samples.loc[:, "delta_p1mi"] = (
                    batch_samples["p1mi"] - control_p1mi
                )

                processed_data.append(batch_samples)

    df_processed = pd.concat(processed_data, ignore_index=True)
    df_processed["pH"] = df_processed["condition"]

    return df_processed
