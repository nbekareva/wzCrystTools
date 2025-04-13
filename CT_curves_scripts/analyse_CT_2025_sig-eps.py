#!/usr/bin/env python
# coding: utf-8

import csv
import glob
import json
import re
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

sys.path.append("/home/nbekareva/TOOLS/utils")


def get_nu_E_substrate(sample):       # # dim-less, 1e9 Pa
    from hexag_cristallo_tool import Wurtzite
    crystal = Wurtzite(3.25, 5.2)
    E, nu = crystal.E_nu_bulk()
    return nu, E


def load_config(json_path, sample, pillar, pillar_dimension_type="initial", test_id=-1):
    """Load configuration from JSON file and return all necessary parameters."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Get basic configuration
    datadir = config["samples"][sample]["datadir"]
    outdir = config["outdir"]
    test_ns = config["samples"][sample][pillar]["tests"]
    date = config["date"]
    frame_stiffness = config["Frame_stiffness"]
    
    # Get pillar dimensions
    dimensions = config["samples"][sample][pillar]["dimensions"][pillar_dimension_type]
    h = dimensions["h"]       # um
    dtop = dimensions["dtop"] # um
    dbott = dimensions["dbott"] # um
    rtop = dtop / 2
    rbott = dbott / 2
    
    # Set output paths
    outfile = Path(outdir) / f'CT_analysis_{sample}.csv'
    out_fig = Path(outdir) / f'CT_C_{sample}_{pillar}.png'
    
    # Print summary
    print(f"Compression test of {date}")
    print(f"Loaded configuration for sample {sample}, pillar {pillar}")
    print(f"Test numbers: {test_ns}")
    print(f"Using {pillar_dimension_type} dimensions: h={h}µm, dtop={dtop}µm, dbott={dbott}µm")
    
    if sample == "0001O":
        test_id = 0
    return {
        'datadir': datadir,
        'outdir': outdir,
        'test_ns': test_ns,
        'date': date,
        'frame_stiffness': frame_stiffness,
        'h': h,
        'rtop': rtop,
        'rbott': rbott,
        'outfile': outfile,
        'out_fig': out_fig,
        'test_id': test_id
    }


def find_test_files(datadir, test_ns):
    """Find test files in the data directory."""
    os.chdir(datadir)
    files = " ".join(glob.glob("*.txt"))
    tests = [re.search(fr"[\d\-_]{{19}}_{n}_0(_Aborted){{0,1}}.txt", files).group(0) for n in test_ns]
    return tests


def load_data(test_file):
    """Load and preprocess data from a test file."""
    data = pd.read_csv(test_file, skiprows=(0,1,3), sep='\t', 
                      encoding='unicode_escape', low_memory=False)
    return data


def process_data(sample, data, frame_stiffness, h, rtop, rbott):
    """Process raw data to calculate strain and stress."""
    # Strip useless data
    data = data[data['Sample Displace'] >= 0]
    data = data.loc[:, ['Phase', 'Force A', 'Sample Displace']].astype(float)
    data.reset_index(inplace=True, drop=True)
    
    # Calculate true displacement, strain, and stress
    nu, E = get_nu_E_substrate(sample)        # dim-less, 1e9 Pa
    data['h_Sneddon_bott'] = (1 - nu**2) * data['Force A'] / (2*E * rbott) * 1e-3    # um
    nu, E = 0.07, 1143      # diamond
    data['h_Sneddon_top'] = (1 - nu**2) / (2*E * rtop) * data['Force A'] * 1e-3    # um
    # correct displacement for frame compliance & Sneddon at the pillar bottom
    data['trueD'] = data['Sample Displace'] - data['Force A'] / frame_stiffness - \
                    data['h_Sneddon_bott'] - data['h_Sneddon_top']
    data['Strain'] = data['trueD'] / h
    data['Stress'] = data['Force A'] / (0.5 * np.pi * (rtop**2 + rbott**2))
    
    return data[['Strain', 'Stress', 'Force A', 'Phase']]


def lin_fit(x, y):
    """Perform linear fit and return fitted values and parameters."""
    x0, x1 = x.values[0], x.values[-1]
    p1, p0 = np.polyfit(x, y, deg=1)
    x_fitted = np.linspace(x0, x1, 10)
    y_fitted = p0 + p1 * x_fitted
    x_int = -p0 / p1
    return x_fitted, y_fitted, p1, x_int


def find_inflection_points(epsilon, sigma, sampling_freq, E):
    """Find flexure points in the stress-strain curve."""
    smooth = gaussian_filter(sigma, 1)
    
    # Calculate derivatives
    epsilon_sampled = epsilon.iloc[::sampling_freq]
    smooth_sampled = smooth[::sampling_freq]
    
    # First derivative
    d = np.diff(smooth_sampled) / np.diff(epsilon_sampled)
    
    # Second derivative
    dd = np.diff(d) / np.diff(epsilon_sampled.iloc[:-1])
    
    # Threshold for detecting changes
    threshold = E * 1000        # 2024: # 40000/5000 for 0001, 1000 for 11-20, 10000/ for 0001O
    change_indices = np.where(dd < -threshold)[0]
    
    # Convert indices back to original data indices
    return change_indices * sampling_freq + epsilon.index[0]


def analyze_compression_test(json_path, sample, pillar, pillar_dimension_type="initial", test_id=-1, create_plot=True):
    """Main function to analyze compression test data."""
    from yield_point_detection import bilinear_fit_yield_point, plot_bilinear_fit
    
    # Load configuration
    config = load_config(json_path, sample, pillar, pillar_dimension_type, test_id)
    
    # Find and load test files
    tests = find_test_files(config['datadir'], config['test_ns'])
    print(f'Compression tests for sample {sample}, pillar {pillar}, date {config["date"]}: \n{tests}')
    
    # Load chosen test
    test_to_load = tests[config['test_id']]
    data = load_data(test_to_load)
    
    # Process data
    data = process_data(sample, data, config['frame_stiffness'], config['h'], config['rtop'], config['rbott'])
    
    # Calculate Young's modulus at withdrawal
    fmax = data.loc[data['Phase'] == 4, 'Force A'].max()
    to_fit_wd = (data['Phase'] == 4) & (0.5*fmax < data['Force A']) & (data['Force A'] < 0.9*fmax)
    epsilon_wd, sigma_wd, E_wd, eps_fin = lin_fit(data.loc[to_fit_wd, 'Strain'], data.loc[to_fit_wd, 'Stress'])
    
    # Calculate Young's modulus at compression
    to_fit_ct = (data['Phase'] == 1) & (0.4*fmax < data['Force A']) & (data['Force A'] < 0.6*fmax)
    epsilon_ct, sigma_ct, E_ct, eps0 = lin_fit(data.loc[to_fit_ct, 'Strain'], data.loc[to_fit_ct, 'Stress'])
    
    # Calculate yield strength
    to_search = (data['Phase'] == 1) & (0.4*fmax < data['Force A']) & (data['Force A'] < 1.0*fmax)
    flex_pts = find_inflection_points(data.loc[to_search, 'Strain'], data.loc[to_search, 'Stress'], 250, E_wd)
    sigma_y_id, eps_y, sigma_y, E_bilinear = bilinear_fit_yield_point(
        data.loc[to_search, 'Strain'], 
        data.loc[to_search, 'Stress'],
        min_points=15  # Adjust based on your data resolution
    )
    
    # Calculate elastic and plastic deformation
    eps_el = eps_y - eps0
    eps_pl = eps_fin - eps0
    
    # Create individual plot if requested
    if create_plot:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original plot
    create_individual_plot(data, epsilon_wd, sigma_wd, epsilon_ct, sigma_ct, eps0, eps_fin, 
                flex_pts, eps_y, sigma_y, config['test_ns'][config['test_id']], None, ax=ax1)
    ax1.set_title(f"Test #{config['test_ns'][config['test_id']]} - Original Method")

    # Bilinear fit plot
    plot_bilinear_fit(data.loc[to_search, 'Strain'], data.loc[to_search, 'Stress'], 
                    sigma_y_id, ax=ax2)
    ax2.set_title(f"Test #{config['test_ns'][config['test_id']]} - Bilinear Fit Method")

    plt.tight_layout()
    plt.savefig(config['out_fig'], dpi=300)
    
    # Print summary
    print_summary(sample, pillar, E_wd, E_ct, eps_fin, eps_el, eps_y, sigma_y, eps_pl)
    
    # Save results
    results = {
        'sample': sample,
        'pillar': pillar,
        'E_wd': E_wd,
        'E_ct': E_ct,
        'eps0': eps0,
        'eps_fin': eps_fin,
        'eps_el': eps_el,
        'eps_pl': eps_pl,
        'eps_y': eps_y,
        'sigma_y': sigma_y,
        'test_n': config['test_ns'][config['test_id']]
    }
    
    # Optional: write to CSV
    # write_results(config['outfile'], results)
    
    return results, data


def create_individual_plot(data, epsilon_wd, sigma_wd, epsilon_ct, sigma_ct, eps0, eps_fin, 
                flex_pts, eps_y, sigma_y, test_n, out_fig, ax=None):
    """Create and save a plot of the stress-strain curve for a single sample-pillar."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points by phase
    for phase in data['Phase'].unique():
        cond = data['Phase'] == phase
        ax.scatter(data.loc[cond, 'Strain'], data.loc[cond, 'Stress'], 
                 s=0.1, alpha=0.5, label=f'Phase {phase}')
    
    # Plot fits and important points
    ax.plot(epsilon_wd, sigma_wd, "--", c='tab:pink', label='Linear fit withdrawal')
    ax.plot(epsilon_ct, sigma_ct, "--", c='tab:purple', label='Linear fit compression')
    ax.scatter([eps0, eps_fin], [0, 0], c='tab:pink', label='eps0, eps_fin')
    
    # Plot flexure points if they exist
    if len(flex_pts) > 0 and all(i in data.index for i in flex_pts):
        ax.scatter(data.loc[flex_pts, 'Strain'], data.loc[flex_pts, 'Stress'], c='y')
    
    # Plot yield strength
    ax.scatter([eps_y, eps_y], [sigma_y, 0], c='cyan', label='Yield strength')
    
    # Format plot
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title(f"Test #{test_n}")
    ax.grid()
    ax.legend()
    
    # Save figure only if we created a new one
    if out_fig is not None and ax is None:
        plt.savefig(out_fig, dpi=300)
    
    return ax


def create_combined_plot(all_results, out_fig_path=None):
    """
    Create a combined plot for multiple sample-pillars.
    
    Parameters:
    -----------
    all_results : list of tuples
        List of (results, data) tuples for each sample-pillar
    out_fig_path : str or Path, optional
        Path to save the combined figure
    """
    plt.figure(figsize=(12, 8))
    
    # Use a colormap for different samples
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(all_results)))
    
    # Plot each sample-pillar with different color
    for i, (results, data) in enumerate(all_results):
        sample = results['sample']
        pillar = results['pillar']
        label = f"{sample}-{pillar}"
        color = colors[i]
        
        # Plot the stress-strain data for Phase 1 (compression)
        comp_data = data[data['Phase'] == 1]
        plt.plot(comp_data['Strain'], comp_data['Stress'], 
                 '-', color=color, linewidth=1, alpha=0.7, label=label)
        
        # Plot withdrawal data
        withdraw_data = data[data['Phase'] == 4]
        if len(withdraw_data) > 0:
            plt.plot(withdraw_data['Strain'], withdraw_data['Stress'],
                    '--', color=color, linewidth=0.7, alpha=0.5)
        
        # Add linear fit lines for compression and withdrawal
        # Calculate start and end points for compression fit line
        eps_start_comp = results['eps0']
        # Find a suitable end point for the compression fit line
        eps_range_comp = comp_data['Strain'].max() - eps_start_comp
        eps_end_comp = eps_start_comp + eps_range_comp * 0.7  # Endpoint at 70% of the range
        
        # Compression fit line
        strain_fit_comp = np.linspace(eps_start_comp, eps_end_comp, 10)
        stress_fit_comp = results['E_ct'] * (strain_fit_comp - eps_start_comp)
        plt.plot(strain_fit_comp, stress_fit_comp, '-.', color=color, 
                 linewidth=1.5, alpha=0.8)
        
        # Withdrawal fit line - fixing to ensure it is plotted correctly
        if len(withdraw_data) > 0:
            # Find the range of withdrawal data that has meaningful stress values
            high_stress_wd = withdraw_data[withdraw_data['Stress'] > withdraw_data['Stress'].max() * 0.1]
            
            if len(high_stress_wd) > 0:
                # Get the strain range for the withdrawal phase (excluding very low stress points)
                min_strain_wd = high_stress_wd['Strain'].min()
                max_strain_wd = high_stress_wd['Strain'].max()
                
                # Create fit line points
                strain_fit_wd = np.linspace(min_strain_wd, max_strain_wd, 10)
                
                # Use the correct formula: stress = E_wd * (strain - eps_fin)
                # This will naturally create a line that eventually hits zero stress at eps_fin
                stress_fit_wd = results['E_wd'] * (strain_fit_wd - results['eps_fin'])
                
                # Plot only if we have valid points
                plt.plot(strain_fit_wd, stress_fit_wd, ':', color=color, 
                         linewidth=1.5, alpha=0.8)
        
        # Mark yield strength
        plt.scatter(results['eps_y'], results['sigma_y'], 
                    color=color, marker='o', s=50, edgecolors='black')
    
    # Format plot
    plt.ylim(-100, 2000)
    plt.xlabel('Strain', fontsize=12)
    plt.ylabel('Stress (MPa)', fontsize=12)
    plt.title("Comparison of Stress-Strain Curves", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend for line styles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', lw=1.5, label='Compression'),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='Withdrawal'),
        Line2D([0], [0], color='gray', linestyle='-.', lw=1.5, label='E_ct fit'),
        Line2D([0], [0], color='gray', linestyle=':', lw=1.5, label='E_wd fit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='Yield point', markeredgecolor='black')
    ]
    
    # Create two legends: one for samples and one for line types
    handles, labels = plt.gca().get_legend_handles_labels()
    first_legend = plt.legend(handles, labels, loc='upper left', fontsize=10, title="Samples")
    plt.gca().add_artist(first_legend)
    second_legend = plt.legend(handles=legend_elements, loc='upper right', fontsize=10, title="Features")
    
    # Add table of properties
    table_data = []
    table_columns = ["Sample-Pillar", "E_ct (MPa)", "E_wd (MPa)", "E_mean (MPa)", "σᵧ (MPa)", "ε_el", "ε_pl"]
    
    for results, _ in all_results:
        E_avg = (results['E_wd'] + results['E_ct']) / 2
        row = [
            f"{results['sample']}-{results['pillar']}", 
            f"{results['E_ct']:.0f}",
            f"{results['E_wd']:.0f}",
            f"{E_avg:.0f}", 
            f"{results['sigma_y']:.0f}", 
            f"{results['eps_el']:.3f}", 
            f"{results['eps_pl']:.3f}"
        ]
        table_data.append(row)
    
    # Add table below the plot
    table = plt.table(
        cellText=table_data,
        colLabels=table_columns,
        loc='bottom',
        bbox=[0.0, -0.4, 1.0, 0.25]  # Adjust to accommodate more columns
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Make more room for the expanded table
    
    # Save figure if path provided
    if out_fig_path:
        plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_summary(sample, pillar, E_wd, E_ct, eps_fin, eps_el, eps_y, sigma_y, eps_pl):
    """Print a summary of the analysis results."""
    print(f"\nSample {sample}, pillar {pillar}:")
    print(f"Young's module: withdrawal {E_wd:,.0f} MPa, compression {E_ct:,.0f} MPa, mean {(E_wd+E_ct)/2:,.0f}")
    print(f"Max plastic deformation, eps_fin:\t {eps_fin:.3f}")
    print(f"Max elastic deformation, eps_el:\t {eps_el:.3f}")
    print(f'Yield strength:\t\t\t\t {eps_y:.3f}, {sigma_y:,.0f} MPa')
    print(f'Total plastic deformation:\t\t {eps_pl:.3f}')


def write_results(outfile, results):
    """Write results to a CSV file."""
    with open(outfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([
            results['sample'], 
            results['pillar'], 
            results['E_wd'], 
            results['E_ct'], 
            results['eps0'], 
            results['eps_fin'], 
            results['eps_el'], 
            results['eps_pl'], 
            results['eps_y'], 
            results['sigma_y']
        ])


def batch_analyze_samples(json_path, sample_pillar_list, combined_fig_path=None, pillar_dimension_type="initial", test_id=-1):
    """
    Analyze multiple sample-pillars and create a combined plot.
    
    Parameters:
    -----------
    json_path : str
        Path to the JSON configuration file
    sample_pillar_list : list of tuples
        List of (sample, pillar) tuples to analyze
    combined_fig_path : str or Path, optional
        Path to save the combined figure
    pillar_dimension_type : str, default="initial"
        Type of pillar dimensions to use
    test_id : int, default=0
        Index of the test to use
    
    Returns:
    --------
    list
        List of (results, data) tuples for each sample-pillar
    """
    all_results = []
    
    # Analyze each sample-pillar
    for sample, pillar in sample_pillar_list:
        print(f"\n{'='*60}\nAnalyzing sample {sample}, pillar {pillar}...\n{'='*60}")
        results, data = analyze_compression_test(
            json_path, sample, pillar, 
            pillar_dimension_type=pillar_dimension_type, 
            test_id=test_id,
            create_plot=True  # Don't create individual plots during batch analysis
        )
        all_results.append((results, data))
    
    # Create combined plot
    if combined_fig_path:
        create_combined_plot(all_results, combined_fig_path)
    
    return all_results


def save_batch_results_to_csv(all_results, csv_path):
    """
    Save the results of multiple sample-pillars to a CSV file.
    
    Parameters:
    -----------
    all_results : list of tuples
        List of (results, data) tuples for each sample-pillar
    csv_path : str or Path
        Path to save the CSV file
    """
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        # Write header
        writer.writerow([
            'Sample', 'Pillar', 'Test', 
            'E_ct (MPa)', 'E_wd (MPa)', 'E_avg (MPa)',
            'eps0', 'eps_fin', 'eps_el', 'eps_pl', 'eps_y', 'sigma_y (MPa)'
        ])
        
        # Write data for each sample-pillar
        for results, _ in all_results:
            writer.writerow([
                results['sample'],
                results['pillar'],
                results['test_n'],
                f"{results['E_ct']:.0f}",
                f"{results['E_wd']:.0f}",
                f"{(results['E_wd'] + results['E_ct'])/2:.0f}",
                f"{results['eps0']:.3f}",
                f"{results['eps_fin']:.3f}",
                f"{results['eps_el']:.3f}",
                f"{results['eps_pl']:.3f}",
                f"{results['eps_y']:.3f}",
                f"{results['sigma_y']:.0f}"
            ])
    
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    # Example usage for multiple sample-pillars
    # json_path = r"/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DATA/FemtoTools/1st_series_pillar_CT_info.json"
    json_path = r"/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DATA/FemtoTools/2nd_series_pillar_CT_info.json"
    
    # Define sample-pillars to analyze
    sample_pillar_list = [
        # ('10-10', 'P2'),
        # ('10-10', 'P3'),
        # ('11-20', 'P4')
        ('0001Zn', 'P5'),
        ('0001Zn', 'P6')
        # ('0001Zn', 'P4'),
        # ('0001Zn', 'P5'),
        # ('0001O', 'P4')
        # Add more sample-pillar combinations as needed
    ]
    
    # Set output paths
    from pathlib import Path
    outdir = Path("/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DataTreatment/FemtoTools")
    outdir.mkdir(exist_ok=True)
    
    combined_fig_path = outdir / "combined_stress_strain.png"
    csv_results_path = outdir / "all_sample_results.csv"
    
    # Run batch analysis
    all_results = batch_analyze_samples(
        json_path, 
        sample_pillar_list, 
        combined_fig_path=combined_fig_path,
        test_id=-1
    )
    
    # Save results to CSV
    save_batch_results_to_csv(all_results, csv_results_path)
    
    # If you want to analyze just a single sample-pillar
    # results, data = analyze_compression_test(json_path, '10-10', 'P1')