"""
Plotting utilities for DNA brush paper analysis
Following CLAUDE.md formatting guidelines
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.optimize import curve_fit
import scipy.stats as stats
import pandas as pd
from typing import Optional, Tuple, List

from utils import defaults


default_figsize = (defaults.fig_width, defaults.fig_height)

# Set random seed for reproducible error bands
np.random.seed(42)


def fit_function(c, delta_d_max, log_c_half):
    """Michaelis-Menten style function: ΔD = ΔD_max * c / (c + c_1/2)"""
    return delta_d_max * c / (c + np.pow(10, log_c_half))


def setup_plot_style():
    """Set up seaborn style according to CLAUDE.md guidelines"""
    sns.set_style("white")
    sns.set_palette("colorblind")


def format_axes(ax):
    """Apply standard formatting to axes according to CLAUDE.md guidelines"""
    # Remove top and right spines, remove grey background
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Add tick marks on all visible sides (bottom and left)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        top=False,
        right=False,
        bottom=True,
        left=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2,
        top=False,
        right=False,
        bottom=True,
        left=True,
    )

    # Ensure ticks are visible
    ax.tick_params(axis="x", which="both", labelbottom=True)
    ax.tick_params(axis="y", which="both", labelleft=True)

    # Center-align x-axis tick labels vertically (midline aligns with tick)
    for label in ax.get_xticklabels():
        label.set_va("center")


def plot_error_ellipse(
    mean, cov_matrix, color, ax=None, x_exponent_base=None, **kwargs
):
    """
    Plot an error ellipse representing the standard error of a 2D variable.

    Parameters:
    -----------
    mean : array-like, shape (2,)
        Mean values [x_mean, y_mean]
    cov_matrix : array-like, shape (2, 2)
        Covariance/error matrix
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    color :
        color in which to plot the ellipse
    x_exponent_base: float, optional
        base for exponential transform of x (if None, no transform occurs)
    **kwargs : dict
        Additional arguments passed to matplotlib.patches.Ellipse

    Returns:
    --------
    ellipse : matplotlib.patches.Ellipse
        The ellipse patch object
    """
    if ax is None:
        ax = plt.gca()

    # Convert to numpy arrays
    mean = np.array(mean)
    cov_matrix = np.array(cov_matrix)

    # Eigendecomposition to get ellipse parameters
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors by eigenvalue magnitude
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    # Semi-axes lengths (scaled by chi-squared value)
    width = np.sqrt(eigenvals[0])
    height = np.sqrt(eigenvals[1])

    # Rotation angle (in degrees)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

    # Default styling
    ellipse_kwargs = {"color": color, "linewidth": 2, "alpha": 0.7, "zorder": -1}
    ellipse_kwargs.update(kwargs)

    # Generate points along the ellipse perimeter
    theta = np.linspace(0, 2 * np.pi, 100)
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))

    # Ellipse in rotated coordinate system
    x_rot = (width / 2) * np.cos(theta)
    y_rot = (height / 2) * np.sin(theta)

    # Rotate back to original coordinate system
    x_ellipse = mean[0] + x_rot * cos_angle - y_rot * sin_angle
    y_ellipse = mean[1] + x_rot * sin_angle + y_rot * cos_angle

    if x_exponent_base is not None:
        x_ellipse = np.pow(x_exponent_base, x_ellipse)

    ax.plot(x_ellipse, y_ellipse, **ellipse_kwargs)

    return


def plot_brush_data_continuous(
    df_data,
    category_col,
    value_col,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    legend_title,
    figsize=(defaults.fig_width * 2, defaults.fig_height),
):
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
    cmap = plt.cm.get_cmap("viridis")
    norm = Normalize(vmin=min(categories), vmax=max(categories))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"][: len(categories)]

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
        ax.scatter(
            data_subset[x_col],
            data_subset[y_col],
            color=color,
            marker=markers[i],
            edgecolors="black",
            linewidths=0.5,
            label=f"{int(category)} bp",
            s=50,
        )

        # Fit and plot curve
        plot_fit_curve(
            ax, data_subset, value_col, y_col, x_col, df_data[value_col].values, color
        )

    # Format plot
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])  # Reverse x-axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False
    )
    ax.set_title(title)

    format_axes(ax)

    plt.tight_layout()
    return fig, ax


def plot_brush_data_categorical(
    df_data,
    category_col,
    value_col,
    x_col,
    y_col,
    ax_1_title,
    ax_2_title,
    xlabel,
    ylabel,
    legend_title,
    figsize=(defaults.fig_width * 2, defaults.fig_height),
):
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
    ax_1_title : str
        Title for first axis
    ax_2_title : str
        Title for second axis
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
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=figsize)

    # Get unique categories and assign colors
    categories = sorted(df_data[category_col].unique())
    colors = sns.color_palette("colorblind", n_colors=len(categories))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"][: len(categories)]

    # Set extended axis limits first for categorical data
    all_ratios = df_data[x_col].values
    min_ratio = min(all_ratios) * 0.3  # Extend further left
    max_ratio = max(all_ratios) * 3.0  # Extend further right
    ax_1.set_xlim(min_ratio, max_ratio)  # Normal order, will be reversed later

    # Plot individual data points and fit curves
    for i, category in enumerate(categories):
        data_subset = df_data[df_data[category_col] == category]
        color = colors[i]

        # Plot scatter points
        ax_1.scatter(
            data_subset[x_col],
            data_subset[y_col],
            color=color,
            marker=markers[i],
            edgecolors="black",
            linewidths=0.5,
            label=f"{category}",
            s=50,
        )

        # Fit and plot curve
        param_mean, param_cov = plot_fit_curve(
            ax_1, data_subset, value_col, y_col, x_col, df_data[value_col].values, color
        )

        if param_mean is None or param_cov is None:
            print("Could not fit curve for category: ", category)
            continue

        # Flip x and y
        param_mean = param_mean[::-1]
        param_cov = param_cov[::-1, ::-1]

        ax_2.scatter(
            [np.pow(10, param_mean[0])],
            [param_mean[1]],
            color=color,
            marker=markers[i],
            edgecolors="black",
            linewidths=0.5,
            label=f"{category}",
            s=50,
        )

        plot_error_ellipse(
            param_mean, param_cov, ax=ax_2, color=color, x_exponent_base=10
        )

    # Format plot
    ax_1.set_xscale("log")
    ax_1.set_xlim(ax_1.get_xlim()[::-1])  # Reverse x-axis
    ax_1.set_xlabel(xlabel)
    ax_1.set_ylabel(ylabel)
    ax_1.legend(
        title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False
    )
    ax_1.set_title(ax_1_title)

    format_axes(ax_1)

    ax_2.set_xlabel("$c_{1/2}$")
    ax_2.set_xscale("log")
    ax_2.set_ylabel(r"$\Delta D_{max} / nm$")
    ax_2.set_title(ax_2_title)

    format_axes(ax_2)

    plt.tight_layout()
    return fig


def plot_fit_curve(
    ax, data_subset, value_col, y_col, x_col, all_concentrations, color
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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

            # Define the bounds
            # Fit function takes c, delta_d_max, c_half
            # Bounds on delta_d_max are [0, 100]
            # Bounds on log_c_half are [-5, -1]
            bounds = [(0, -5), (100, -1)]

            # Fit the function with better initial parameters
            popt, pcov = curve_fit(
                fit_function, c_sorted, y_sorted, bounds=bounds, maxfev=5000
            )

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

            param_samples = np.random.multivariate_normal(popt, pcov, n_samples)

            for i in range(n_samples):
                # Sample parameters from their distributions
                delta_d_max_sample, c_half_sample = param_samples[i]

                # Calculate curve for this sample
                y_sample = fit_function(c_smooth, delta_d_max_sample, c_half_sample)
                y_samples.append(y_sample)

            y_samples = np.array(y_samples)
            y_lower = np.percentile(y_samples, 16, axis=0)  # -1 sigma
            y_upper = np.percentile(y_samples, 84, axis=0)  # +1 sigma

            # Plot fitted curve
            ax.plot(ratio_smooth, y_fit, color=color, linewidth=2, alpha=0.8)

            # Plot error band
            ax.fill_between(ratio_smooth, y_lower, y_upper, color=color, alpha=0.3)

            return popt, pcov

        except (RuntimeError, ValueError) as e:
            print(f"Could not fit curve: {e}")
            return None, None

    return None, None


def plot_linear_relationship(
    df_data, x_col, y_col, title, xlabel, ylabel, figsize=default_figsize, color_index=0
):
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
    color_index : int
        Index of color to use from colorblind palette (default 0)

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
    linregress_result = stats.linregress(x_data, y_data)
    slope = linregress_result.slope  # type: ignore
    intercept = linregress_result.intercept  # type: ignore
    r_value = linregress_result.rvalue  # type: ignore
    p_value = linregress_result.pvalue  # type: ignore
    std_err = linregress_result.stderr  # type: ignore

    # Use colorblind palette for consistency
    colors = sns.color_palette("colorblind")
    plot_color = colors[color_index]  # Use specified color from colorblind palette

    # Plot scatter points
    ax.scatter(
        x_data,
        y_data,
        color=plot_color,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        s=50,
        alpha=0.7,
    )

    # Set axis limits first
    x_margin = (max(x_data) - min(x_data)) * 0.05  # 5% margin
    ax.set_xlim(min(x_data) - x_margin, max(x_data) + x_margin)

    # Plot regression line in same color - extend to full axis limits
    ax_xlim = ax.get_xlim()
    x_line = np.linspace(ax_xlim[0], ax_xlim[1], 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=plot_color, linewidth=2, alpha=0.8)

    # Calculate confidence interval for regression line
    def prediction_interval(x, y, new_x, confidence=0.68):
        """Calculate prediction interval for linear regression"""
        n = len(x)
        x_mean = np.mean(x)

        # Calculate residual sum of squares
        y_pred = slope * x + intercept
        rss = np.sum((y - y_pred) ** 2)
        mse = rss / (n - 2)

        # Calculate standard error of prediction
        se_pred = np.sqrt(
            mse * (1 + 1 / n + (new_x - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
        )

        # t-statistic for confidence interval
        t_val = stats.t.ppf((1 + confidence) / 2, n - 2)

        return t_val * se_pred

    # Plot confidence band in same color
    y_pred = slope * x_line + intercept
    ci = prediction_interval(x_data, y_data, x_line)
    ax.fill_between(x_line, y_pred - ci, y_pred + ci, color=plot_color, alpha=0.3)

    # Add regression statistics to plot
    r_squared = r_value**2
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    stats_text = f"R² = {r_squared:.3f}\n{p_text}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

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


def plot_calcein_release(
    experiment_names: List[str],
    mean_df: pd.DataFrame,
    sem_df: pd.DataFrame,
    raw_release_df: Optional[pd.DataFrame] = None,
    group_size: Optional[int] = 4,
):
    """
    Plot calcein release data with both bar chart (final timepoint) and line chart (full timecourse).

    Args:
        mean_df: DataFrame with mean release values over time
        sem_df: DataFrame with standard error values over time
        raw_release_df:
        experiment_names: List of experiment names to use as labels
        group_size: Number of samples per group for color cycling
    """
    # Set up the plot style
    setup_plot_style()

    # Create figure with two subplots
    fig1, ax1 = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    fig2, ax2 = plt.subplots(figsize=(defaults.fig_width * 1.5, defaults.fig_height))

    colors = sns.color_palette("colorblind", n_colors=group_size)

    final_means = mean_df.iloc[-1].astype(float)
    final_sems = sem_df.iloc[-1].astype(float)
    # Convert to float - handle various index types
    # Extract the actual value from the index
    index_values = mean_df.index.to_numpy()
    final_time_value = index_values[-1]
    if isinstance(final_time_value, (int, float, np.integer, np.floating)):
        final_time = float(final_time_value)
    else:
        # For other types like strings, try to convert
        final_time = float(final_time_value)

    # Group bars by color
    for i, mean_val in enumerate(final_means):
        color_idx = i if group_size is None else i % group_size
        color = colors[color_idx]

        if raw_release_df is None:
            yerr = final_sems.iloc[i]
        else:
            yerr = None

        ax1.bar(i, mean_val, yerr=yerr, color=color, capsize=3, edgecolor="none")

        if raw_release_df is not None:
            well_column_on_plate = mean_df.columns[i]
            wells_in_column = list(
                filter(
                    lambda x: str(well_column_on_plate) in str(x),
                    raw_release_df.columns,
                )
            )

            raw_values = raw_release_df[wells_in_column].iloc[-1].astype(float)

            # Scatter raw values with horizontal jitter
            n_points = len(raw_values)
            if n_points > 0:
                # Spread points within ±0.15 of the bar center
                jitter = np.linspace(-0.15, 0.15, n_points)
                ax1.scatter(
                    i + jitter,
                    raw_values,
                    marker="o",
                    color=color,
                    edgecolor="black",
                    alpha=1,
                    s=18,
                    zorder=10,
                    linewidths=1,
                )

    # Add a horizontal line at y=0
    ax1.axhline(0, color="black", linewidth=1, zorder=5)

    ax1.set_xlabel("Well")
    ax1.set_ylabel("Calcein Release (%)")
    ax1.set_title(f"Final Release at {final_time:.1f} min")
    ax1.set_xticks(np.arange(len(experiment_names)))
    ax1.set_xticklabels(experiment_names, rotation=90, ha="right")
    format_axes(ax1)

    # Create line chart with shaded areas
    time_points = mean_df.index

    for i, well in enumerate(mean_df.columns):
        color_idx = i if group_size is None else i % group_size
        color = colors[color_idx]

        # Use different alpha and linestyle for different groups
        linestyle = "-" if group_size is None or i < group_size else "--"

        # Plot mean line
        ax2.plot(
            time_points,
            mean_df[well],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=experiment_names[i],
        )

        # Plot shaded error area
        ax2.fill_between(
            time_points,
            mean_df[well] - sem_df[well],
            mean_df[well] + sem_df[well],
            color=color,
            alpha=0.3,
        )

    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Calcein Release (%)")
    ax2.set_title("Release Over Time")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    format_axes(ax2)

    plt.tight_layout()
    return fig1, fig2


def save_plot(fig, filename_base):
    """Save plot as SVG according to CLAUDE.md guidelines"""
    fig.savefig(f"{filename_base}.svg", bbox_inches="tight")
