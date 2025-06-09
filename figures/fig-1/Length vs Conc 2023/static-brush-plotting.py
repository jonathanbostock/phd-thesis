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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.plotting import fit_function, setup_plot_style, format_axes, plot_fit_curve

def main() -> None:
    setup_plot_style()
    
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

    # Create figure with subplots for both plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Get unique brush lengths and create gradient colormap
    brush_lengths = sorted(df_data["Brush Length / bp"].unique())
    # Create a colormap for gradient colors based on brush length
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(brush_lengths), vmax=max(brush_lengths))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(brush_lengths)]
    
    # Store fitted parameters for second plot
    fitted_params = []
    
    # Set axis limits first to ensure curves extend to full range
    all_ratios = df_data["Lipid:DNA Ratio"].values
    min_ratio = min(all_ratios) * 0.3  # Extend further left  
    max_ratio = max(all_ratios) * 3.0  # Extend further right
    ax1.set_xlim(max_ratio, min_ratio)  # Reversed for log scale
    
    # Plot individual data points and fit curves
    for i, brush_length in enumerate(brush_lengths):
        data_subset = df_data[df_data["Brush Length / bp"] == brush_length]
        color = cmap(norm(brush_length))
        ax1.scatter(data_subset["Lipid:DNA Ratio"], data_subset["Delta D"], 
                  color=color, marker=markers[i], edgecolors='black', linewidths=0.5,
                  label=f'{int(brush_length)} bp', s=50)
        
        # Use shared utility for fitting and plotting
        popt, param_errors = plot_fit_curve(ax1, data_subset, "Concentration", "Delta D", 
                                          "Lipid:DNA Ratio", df_data["Concentration"].values, color)
        
        # Store fitted parameters for second plot
        if popt is not None:
            fitted_params.append({
                'brush_length': brush_length,
                'delta_d_max': popt[0],
                'delta_d_max_error': param_errors[0],
                'c_half': popt[1],
                'color': color
            })
    
    # Format first subplot (original plot)
    ax1.set_xscale("log")
    ax1.set_xlabel("Lipid:DNA Ratio")
    ax1.set_ylabel("Delta D")
    ax1.legend(title="Brush Length", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax1.set_title("Static Brush Delta D vs Lipid:DNA Ratio")
    format_axes(ax1)
    
    # Create second subplot: ΔD max vs Brush Length
    if fitted_params:
        # Extract data for plotting
        brush_lengths_fit = [p['brush_length'] for p in fitted_params]
        delta_d_max_vals = [p['delta_d_max'] for p in fitted_params]
        delta_d_max_errors = [p['delta_d_max_error'] for p in fitted_params]
        colors = [p['color'] for p in fitted_params]
        
        # Plot with error bars
        for i, (bl, ddm, err, color) in enumerate(zip(brush_lengths_fit, delta_d_max_vals, delta_d_max_errors, colors)):
            ax2.errorbar(bl, ddm, yerr=err, marker=markers[i], color=color, 
                        markeredgecolor='black', markeredgewidth=0.5, markersize=8,
                        capsize=3, capthick=1, linewidth=0)
        
        # Fit a line through the origin
        # Force intercept to be 0 by fitting y = mx model
        x_data = np.array(brush_lengths_fit)
        y_data = np.array(delta_d_max_vals)
        
        # Fit slope (forcing through origin)
        slope = np.sum(x_data * y_data) / np.sum(x_data**2)
        
        # Plot fitted line
        x_fit = np.linspace(0, max(brush_lengths_fit) * 1.1, 100)
        y_fit = slope * x_fit
        ax2.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.7, label=f'Slope = {slope:.3f}')
        
        # Format second subplot
        ax2.set_xlabel("Brush Length (bp)")
        ax2.set_ylabel("ΔD max")
        ax2.set_title("Fitted ΔD max vs Brush Length")
        ax2.legend(frameon=False)
        
        # Set origin at (0,0)
        ax2.set_xlim(0, max(brush_lengths_fit) * 1.1)
        ax2.set_ylim(0, max(delta_d_max_vals) * 1.1)
        format_axes(ax2)
    
    plt.tight_layout()
    plt.savefig("Static Brush Plot.svg", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
