"""
Plotting utilities for DNA brush paper analysis
Following CLAUDE.md formatting guidelines
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats

# Set random seed for reproducible error bands
np.random.seed(42)


def fit_function(c, delta_d_max, c_half):
    """Michaelis-Menten style function: ΔD = ΔD_max * c / (c + c_1/2)"""
    return delta_d_max * c / (c + c_half)


def setup_plot_style():
    """Set up seaborn style according to CLAUDE.md guidelines"""
    sns.set_style("white")
    sns.set_palette("colorblind")


def format_axes(ax):
    """Apply standard formatting to axes according to CLAUDE.md guidelines"""
    # Remove top and right spines, remove grey background
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Add tick marks on all visible sides (bottom and left)
    ax.tick_params(axis='both', which='major', direction='in', length=4, 
                   top=False, right=False, bottom=True, left=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=2,
                   top=False, right=False, bottom=True, left=True)
    
    # Ensure ticks are visible
    ax.tick_params(axis='x', which='both', labelbottom=True)
    ax.tick_params(axis='y', which='both', labelleft=True)


def plot_brush_data_continuous(df_data, category_col, value_col, x_col, y_col, 
                              title, xlabel, ylabel, legend_title, figsize=(6, 4)):
    """
    Plot brush data with continuous categories using gradient colors
    
    Parameters:
    -----------
    df_data : pandas.DataFrame
        The processed data
    category_col : str
        Column name for the continuous category (e.g., "Brush Length / bp")
    value_col : str
        Column name for the concentration values
    x_col : str
        Column name for x-axis data (e.g., "Lipid:DNA Ratio")
    y_col : str
        Column name for y-axis data (e.g., "Delta D")
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    legend_title : str
        Legend title
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique categories and create gradient colormap
    categories = sorted(df_data[category_col].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(categories), vmax=max(categories))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(categories)]
    
    # Set extended axis limits first for continuous data
    all_ratios = df_data[x_col].values
    min_ratio = min(all_ratios) * 0.3  # Extend further left  
    max_ratio = max(all_ratios) * 3.0  # Extend further right
    ax.set_xlim(min_ratio, max_ratio)  # Normal order, will be reversed later
    
    # Plot individual data points and fit curves
    for i, category in enumerate(categories):
        data_subset = df_data[df_data[category_col] == category]
        color = cmap(norm(category))
        
        # Plot scatter points
        ax.scatter(data_subset[x_col], data_subset[y_col], 
                  color=color, marker=markers[i], edgecolors='black', linewidths=0.5,
                  label=f'{int(category)} bp', s=50)
        
        # Fit and plot curve
        plot_fit_curve(ax, data_subset, value_col, y_col, x_col, 
                      df_data[value_col].values, color)
    
    # Format plot
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # Reverse x-axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.set_title(title)
    
    format_axes(ax)
    
    plt.tight_layout()
    return fig, ax


def plot_brush_data_categorical(df_data, category_col, value_col, x_col, y_col, 
                               title, xlabel, ylabel, legend_title, figsize=(6, 4)):
    """
    Plot brush data with categorical categories using colorblind palette
    
    Parameters:
    -----------
    df_data : pandas.DataFrame
        The processed data
    category_col : str
        Column name for the categorical category (e.g., "Shape")
    value_col : str
        Column name for the concentration values
    x_col : str
        Column name for x-axis data (e.g., "Lipid:DNA Ratio")
    y_col : str
        Column name for y-axis data (e.g., "Delta D")
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    legend_title : str
        Legend title
    figsize : tuple
        Figure size (width, height)
    """
    setup_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique categories and assign colors
    categories = sorted(df_data[category_col].unique())
    colors = sns.color_palette("colorblind", n_colors=len(categories))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(categories)]
    
    # Set extended axis limits first for categorical data
    all_ratios = df_data[x_col].values
    min_ratio = min(all_ratios) * 0.3  # Extend further left  
    max_ratio = max(all_ratios) * 3.0  # Extend further right
    ax.set_xlim(min_ratio, max_ratio)  # Normal order, will be reversed later
    
    # Plot individual data points and fit curves
    for i, category in enumerate(categories):
        data_subset = df_data[df_data[category_col] == category]
        color = colors[i]
        
        # Plot scatter points
        ax.scatter(data_subset[x_col], data_subset[y_col], 
                  color=color, marker=markers[i], edgecolors='black', linewidths=0.5,
                  label=f'{category}', s=50)
        
        # Fit and plot curve
        plot_fit_curve(ax, data_subset, value_col, y_col, x_col, 
                      df_data[value_col].values, color)
    
    # Format plot
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # Reverse x-axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.set_title(title)
    
    format_axes(ax)
    
    plt.tight_layout()
    return fig, ax


def plot_fit_curve(ax, data_subset, value_col, y_col, x_col, all_concentrations, color):
    """
    Fit and plot curve for a data subset with proper uncertainty propagation
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data_subset : pandas.DataFrame
        Data subset for this category
    value_col : str
        Column name for concentration values
    y_col : str
        Column name for y-axis data
    x_col : str
        Column name for x-axis data
    all_concentrations : array
        All concentration values (for determining plot range)
    color : color
        Color for the curve
    """
    c_data = data_subset[value_col].values
    y_data = data_subset[y_col].values
    
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
            
            # Get current axis limits to extend curves to full range
            current_xlim = ax.get_xlim()
            if current_xlim == (0.0, 1.0):  # Default limits, not set yet
                # Use extended range beyond data
                all_ratios = 1 / all_concentrations
                min_ratio = min(all_ratios) * 0.3  # Extend further left
                max_ratio = max(all_ratios) * 3.0  # Extend further right
            else:
                # Use actual axis limits (may be reversed for log scale)
                max_ratio = max(current_xlim)
                min_ratio = min(current_xlim)
            
            # Ensure positive values for log scale
            min_ratio = max(min_ratio, 1e-10)
            max_ratio = max(max_ratio, min_ratio * 10)
            
            # Generate smooth curve covering full axis range
            ratio_smooth = np.logspace(np.log10(min_ratio), np.log10(max_ratio), 200)
            c_smooth = 1 / ratio_smooth
            
            y_fit = fit_function(c_smooth, *popt)
            
            # Properly propagate uncertainty using Monte Carlo approach
            n_samples = 1000
            y_samples = []
            
            for _ in range(n_samples):
                # Sample parameters from their distributions
                delta_d_max_sample = np.random.normal(popt[0], param_errors[0])
                c_half_sample = np.random.normal(popt[1], param_errors[1])
                
                # Calculate curve for this sample
                y_sample = fit_function(c_smooth, delta_d_max_sample, c_half_sample)
                y_samples.append(y_sample)
            
            y_samples = np.array(y_samples)
            y_lower = np.percentile(y_samples, 16, axis=0)  # -1 sigma
            y_upper = np.percentile(y_samples, 84, axis=0)  # +1 sigma
            
            # Plot fitted curve
            ax.plot(ratio_smooth, y_fit, color=color, linewidth=2, alpha=0.8)
            
            # Plot error band
            ax.fill_between(ratio_smooth, y_lower, y_upper, color=color, alpha=0.2)
            
            return popt, param_errors
            
        except (RuntimeError, ValueError) as e:
            print(f"Could not fit curve: {e}")
            return None, None
    
    return None, None


def plot_linear_relationship(df_data, x_col, y_col, title, xlabel, ylabel, figsize=(6, 4)):
    """
    Plot linear relationship with regression line and p-value
    
    Parameters:
    -----------
    df_data : pandas.DataFrame
        The processed data
    x_col : str
        Column name for x-axis data
    y_col : str
        Column name for y-axis data  
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    tuple
        (fig, ax, slope, intercept, r_value, p_value, std_err)
    """
    setup_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get data
    x_data = df_data[x_col].values
    y_data = df_data[y_col].values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Use colorblind palette for consistency
    colors = sns.color_palette("colorblind")
    plot_color = colors[0]  # Use first color from colorblind palette
    
    # Plot scatter points
    ax.scatter(x_data, y_data, color=plot_color, marker='o', 
              edgecolors='black', linewidths=0.5, s=50, alpha=0.7)
    
    # Set axis limits first
    x_margin = (max(x_data) - min(x_data)) * 0.05  # 5% margin
    ax.set_xlim(min(x_data) - x_margin, max(x_data) + x_margin)
    
    # Plot regression line in same color - extend to full axis limits
    ax_xlim = ax.get_xlim()
    x_line = np.linspace(ax_xlim[0], ax_xlim[1], 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=plot_color, linewidth=2, alpha=0.8)
    
    # Calculate confidence interval for regression line
    def prediction_interval(x, y, new_x, confidence=0.95):
        """Calculate prediction interval for linear regression"""
        n = len(x)
        x_mean = np.mean(x)
        
        # Calculate residual sum of squares
        y_pred = slope * x + intercept
        rss = np.sum((y - y_pred) ** 2)
        mse = rss / (n - 2)
        
        # Calculate standard error of prediction
        se_pred = np.sqrt(mse * (1 + 1/n + (new_x - x_mean)**2 / np.sum((x - x_mean)**2)))
        
        # t-statistic for confidence interval
        t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
        
        return t_val * se_pred
    
    # Plot confidence band in same color
    y_pred = slope * x_line + intercept
    ci = prediction_interval(x_data, y_data, x_line)
    ax.fill_between(x_line, y_pred - ci, y_pred + ci, color=plot_color, alpha=0.2)
    
    # Add regression statistics to plot
    r_squared = r_value ** 2
    p_text = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    stats_text = f"R² = {r_squared:.3f}\n{p_text}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-axis to start from 0 for Delta D plots
    if "Δ" in ylabel or "Delta" in ylabel:
        current_ylim = ax.get_ylim()
        ax.set_ylim(0, current_ylim[1])
    
    format_axes(ax)
    
    plt.tight_layout()
    return fig, ax, slope, intercept, r_value, p_value, std_err


def save_plot(fig, filename_base):
    """Save plot as SVG according to CLAUDE.md guidelines"""
    fig.savefig(f"{filename_base}.svg", bbox_inches='tight')