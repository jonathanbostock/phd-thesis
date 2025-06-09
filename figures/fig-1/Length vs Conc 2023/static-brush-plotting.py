# Jonathan Bostock
# 26-Apr-2023
# Analysing the data for the static, dsDNA brushes

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from itertools import product

def fit_function(c, delta_d_max, c_half):
    """Michaelis-Menten style function: ΔD = ΔD_max * c / (c + c_1/2)"""
    return delta_d_max * c / (c + c_half)

def main() -> None:
    # Set seaborn style and colorblind palette
    sns.set_style("white")
    sns.set_palette("colorblind")
    
    df = pd.read_csv("static-brush-data.csv")

    controls = (df["sample_type"] == "ctrl")

    def process_row(row, controls):
        control_data = controls[controls["batch"] == row["batch"]]
        subtract_columns = [c not in ["batch", "sample_type", "sample_number"]
                            for c in control_data.columns]
        control_row = control_data.loc[..., subtract_columns].mean()

        new_row = row.copy()
        new_row[subtract_columns] = new_row[subtract_columns] - control_row
        return new_row

    df_ctrl = df[controls]
    df_data = df[~controls].apply(
        lambda row: process_row(row, df_ctrl),
        axis=1)

    # 5 microliters of 0.5 mg/mL POPC (molar mass 760)
    lipid_moles = 5e-6 * 0.5 / 760
    # microliters of 5 micromolar DNA
    df_data["dna_moles"] = df_data["sample_number"] * 1e-6 * 5e-6
    df_data["Brush Length / bp"] = list(map(float, df_data["sample_type"]))
    df_data["Lipid:DNA Ratio"] = (lipid_moles/df_data["dna_moles"]).astype(int)
    df_data["Concentration"] = 1 / df_data["Lipid:DNA Ratio"]  # Concentration is inverse of ratio
    df_data["Delta D"] = df_data["peak_1_mean_intensity"]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Get unique brush lengths and create gradient colormap
    brush_lengths = sorted(df_data["Brush Length / bp"].unique())
    # Create a colormap for gradient colors based on brush length
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(brush_lengths), vmax=max(brush_lengths))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(brush_lengths)]
    
    # Plot individual data points as scatter
    for i, brush_length in enumerate(brush_lengths):
        data_subset = df_data[df_data["Brush Length / bp"] == brush_length]
        color = cmap(norm(brush_length))
        ax.scatter(data_subset["Lipid:DNA Ratio"], data_subset["Delta D"], 
                  color=color, marker=markers[i], edgecolors='black', linewidths=0.5,
                  label=f'{int(brush_length)} bp', s=50)
        
        # Fit curve for this brush length using concentration (1/ratio)
        c_data = data_subset["Concentration"].values
        y_data = data_subset["Delta D"].values
        
        if len(c_data) > 2:  # Need at least 3 points for fitting
            try:
                # Sort by concentration for smooth curve plotting
                sort_idx = np.argsort(c_data)
                c_sorted = c_data[sort_idx]
                y_sorted = y_data[sort_idx]
                
                # Fit the function with better initial parameters
                popt, pcov = curve_fit(fit_function, c_sorted, y_sorted, 
                                     p0=[max(y_sorted), np.median(c_sorted)], maxfev=5000)
                
                # Calculate parameter errors
                param_errors = np.sqrt(np.diag(pcov))
                
                # Generate smooth curve for plotting (in concentration space)
                # Use full range of all data, not just this brush length
                all_concentrations = df_data["Concentration"].values
                c_smooth = np.logspace(np.log10(min(all_concentrations)), 
                                     np.log10(max(all_concentrations)), 100)
                y_fit = fit_function(c_smooth, *popt)
                
                # Calculate confidence band using parameter errors
                y_upper = fit_function(c_smooth, popt[0] + param_errors[0], 
                                     popt[1] + param_errors[1])
                y_lower = fit_function(c_smooth, popt[0] - param_errors[0], 
                                     popt[1] - param_errors[1])
                
                # Convert back to Lipid:DNA ratio for plotting
                ratio_smooth = 1 / c_smooth
                
                # Plot fitted curve
                ax.plot(ratio_smooth, y_fit, color=color, linewidth=2, alpha=0.8)
                
                # Plot error band
                ax.fill_between(ratio_smooth, y_lower, y_upper, color=color, alpha=0.2)
                
            except (RuntimeError, ValueError) as e:
                print(f"Could not fit curve for {brush_length} bp: {e}")
    
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # Reverse x-axis
    ax.set_xlabel("Lipid:DNA Ratio")
    ax.set_ylabel("Delta D")
    ax.legend(title="Brush Length", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.set_title("Static Brush Delta D vs Lipid:DNA Ratio")
    
    # Remove top and right spines, remove grey background
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Add tick marks
    ax.tick_params(axis='both', which='major', direction='in', length=4)
    ax.tick_params(axis='both', which='minor', direction='in', length=2)
    
    plt.tight_layout()
    plt.savefig("Static Brush Plot.svg", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
