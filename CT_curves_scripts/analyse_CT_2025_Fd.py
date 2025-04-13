#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import csv
import glob
import json
import re
import os
from pathlib import Path

def k_to_E(k, h, rtop, rbott):
    A = 0.5 * np.pi * (rtop**2 + rbott**2)
    E = k * h / A
    return E


def load_config(json_path, sample, pillar, pillar_dimension_type="initial", test_id=0):
    """Load configuration from JSON file and return all necessary parameters."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Get basic configuration
    datadir = config["samples"][sample]["datadir"]
    outdir = config["outdir"]
    test_ns = config["samples"][sample][pillar]["tests"]
    date = config["date"]
    frame_stiffness = config["Frame_stiffness"]
    
    # Get pillar dimensions (for reference only in force-displacement analysis)
    dimensions = config["samples"][sample][pillar]["dimensions"][pillar_dimension_type]
    h = dimensions["h"]       # um
    dtop = dimensions["dtop"] # um
    dbott = dimensions["dbott"] # um
    rtop = dtop / 2
    rbott = dbott / 2
    
    # Set output paths
    outfile = Path(outdir) / f'FD_analysis_{sample}.csv'
    out_fig = Path(outdir) / f'FD_{sample}_{pillar}.png'
    
    # Print summary
    print(f"Compression test of {date}")
    print(f"Loaded configuration for sample {sample}, pillar {pillar}")
    print(f"Test numbers: {test_ns}")
    print(f"Using {pillar_dimension_type} dimensions: h={h}µm, dtop={dtop}µm, dbott={dbott}µm")
    
    # if sample == "0001O":
    #     test_id = 0
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


def process_data(data, frame_stiffness):
    """Process raw data to focus on force and displacement."""
    # Strip useless data
    data = data[data['Sample Displace'] >= 0]
    data = data.loc[:, ['Phase', 'Force A', 'Sample Displace']].astype(float)
    data.reset_index(inplace=True, drop=True)
    
    # Calculate true displacement (corrected for machine compliance)
    data['trueD'] = data['Sample Displace'] - data['Force A'] / frame_stiffness
    
    return data


def lin_fit(x, y):
    """Perform linear fit and return fitted values and parameters."""
    x0, x1 = x.values[0], x.values[-1]
    p1, p0 = np.polyfit(x, y, deg=1)
    x_fitted = np.linspace(x0, x1, 10)
    y_fitted = p0 + p1 * x_fitted
    x_int = -p0 / p1
    return x_fitted, y_fitted, p1, x_int


def find_flexure_points(disp, force, sampling_freq, stiffness):
    """Find flexure points in the force-displacement curve."""
    smooth = gaussian_filter(force, 1)
    
    # Calculate derivatives
    disp_sampled = disp.iloc[::sampling_freq]
    smooth_sampled = smooth[::sampling_freq]
    
    # First derivative
    d = np.diff(smooth_sampled) / np.diff(disp_sampled)
    
    # Second derivative
    dd = np.diff(d) / np.diff(disp_sampled.iloc[:-1])
    
    # Threshold for detecting changes
    threshold = stiffness * 1000  # Adjust threshold for force-displacement
    change_indices = np.where(dd < -threshold)[0]
    
    # Convert indices back to original data indices
    return change_indices * sampling_freq + disp.index[0]


def analyze_force_displacement(json_path, sample, pillar, pillar_dimension_type="initial", test_id=0, create_plot=True):
    """Main function to analyze force-displacement data."""
    # Load configuration
    config = load_config(json_path, sample, pillar, pillar_dimension_type, test_id)
    
    # Find and load test files
    tests = find_test_files(config['datadir'], config['test_ns'])
    print(f'Compression tests for sample {sample}, pillar {pillar}, date {config["date"]}: \n{tests}')
    
    # Load chosen test
    test_to_load = tests[config['test_id']]
    data = load_data(test_to_load)
    
    # Process data
    data = process_data(data, config['frame_stiffness'])
    
    # Calculate stiffness & Young modulus at withdrawal (unloading)
    fmax = data.loc[data['Phase'] == 4, 'Force A'].max()
    to_fit_wd = (data['Phase'] == 4) & (0.3*fmax < data['Force A']) & (data['Force A'] < 0.6*fmax)
    disp_wd, force_wd, k_wd, disp_fin = lin_fit(data.loc[to_fit_wd, 'trueD'], data.loc[to_fit_wd, 'Force A'])
    E_wd = k_to_E(k_wd, h=config["h"], rtop=config["rtop"], rbott=config["rbott"])

    # Calculate stiffness & Young modulus at compression (loading)
    to_fit_ct = (data['Phase'] == 1) & (0.4*fmax < data['Force A']) & (data['Force A'] < 0.5*fmax)
    disp_ct, force_ct, k_ct, disp0 = lin_fit(data.loc[to_fit_ct, 'trueD'], data.loc[to_fit_ct, 'Force A'])
    E_ct = k_to_E(k_ct, h=config["h"], rtop=config["rtop"], rbott=config["rbott"])
    
    # Calculate yield point
    to_search = (data['Phase'] == 1) & (0.4*fmax < data['Force A']) & (data['Force A'] < 1.1*fmax)
    flex_pts = find_flexure_points(data.loc[to_search, 'trueD'], data.loc[to_search, 'Force A'], 250, k_wd)
    
    # Choose yield point (using second flexion point)
    if len(flex_pts) > 1:
        f_y_id = flex_pts[1]
        f_y = data.loc[f_y_id, 'Force A']
        disp_y = data.loc[f_y_id, 'trueD']
    else:
        # Fallback if not enough flexion points found
        f_y_id = data[data['Phase'] == 1]['Force A'].idxmax()
        f_y = data.loc[f_y_id, 'Force A']
        disp_y = data.loc[f_y_id, 'trueD']
    
    # Calculate elastic and plastic displacement
    disp_el = disp_y - disp0
    disp_pl = disp_fin - disp0
    
    # Create individual plot if requested
    if create_plot:
        create_individual_plot(data, disp_wd, force_wd, disp_ct, force_ct, disp0, disp_fin, 
                    flex_pts, disp_y, f_y, config['test_ns'][config['test_id']], config['out_fig'])
    
    # Print summary
    print_summary(sample, pillar, k_wd, k_ct, disp_fin, disp_el, disp_y, f_y, disp_pl)
    
    # Save results
    results = {
        'sample': sample,
        'pillar': pillar,
        'k_wd': k_wd,
        'k_ct': k_ct,
        'E_wd': E_wd,
        'E_ct': E_ct,
        'disp0': disp0,
        'disp_fin': disp_fin,
        'disp_el': disp_el,
        'disp_pl': disp_pl,
        'disp_y': disp_y,
        'f_y': f_y,
        'test_n': config['test_ns'][config['test_id']]
    }
    
    return results, data


def create_individual_plot(data, disp_wd, force_wd, disp_ct, force_ct, disp0, disp_fin, 
                flex_pts, disp_y, f_y, test_n, out_fig):
    """Create and save a plot of the force-displacement curve for a single sample-pillar."""
    plt.figure(figsize=(10, 6))
    
    # Plot data points by phase
    for phase in data['Phase'].unique():
        cond = data['Phase'] == phase
        plt.scatter(data.loc[cond, 'trueD'], data.loc[cond, 'Force A'], 
                   s=0.1, alpha=0.5, label=f'Phase {phase}')
    
    # Plot fits and important points
    plt.plot(disp_wd, force_wd, "--", c='tab:pink', label='Linear fit withdrawal')
    plt.plot(disp_ct, force_ct, "--", c='tab:purple', label='Linear fit compression')
    plt.scatter([disp0, disp_fin], [0, 0], c='tab:pink', label='disp0, disp_fin')
    
    # Plot flexure points if they exist
    if len(flex_pts) > 0 and all(i in data.index for i in flex_pts):
        plt.scatter(data.loc[flex_pts, 'trueD'], data.loc[flex_pts, 'Force A'], c='y')
    
    # Plot yield point
    plt.scatter([disp_y, disp_y], [f_y, 0], c='cyan', label='Yield point')
    
    # Format plot
    plt.xlabel('Displacement (µm)')
    plt.ylabel('Force (µN)')
    plt.title(f"Test #{test_n}")
    plt.grid()
    plt.legend()
    
    # Save figure
    plt.savefig(out_fig, dpi=300)
    plt.show()


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
        
        # Plot the force-displacement data for Phase 1 (compression)
        comp_data = data[data['Phase'] == 1]
        plt.plot(comp_data['trueD'], comp_data['Force A'], 
                 '-', color=color, linewidth=1, alpha=0.7, label=label)
        
        # Plot withdrawal data
        withdraw_data = data[data['Phase'] == 4]
        if len(withdraw_data) > 0:
            plt.plot(withdraw_data['trueD'], withdraw_data['Force A'],
                    '--', color=color, linewidth=0.7, alpha=0.5)
        
        # Add linear fit lines for compression and withdrawal
        # Calculate start and end points for compression fit line
        disp_start_comp = results['disp0']
        # Find a suitable end point for the compression fit line
        disp_range_comp = comp_data['trueD'].max() - disp_start_comp
        disp_end_comp = disp_start_comp + disp_range_comp * 0.7  # Endpoint at 70% of the range
        
        # Compression fit line
        disp_fit_comp = np.linspace(disp_start_comp, disp_end_comp, 10)
        force_fit_comp = results['k_ct'] * (disp_fit_comp - disp_start_comp)
        plt.plot(disp_fit_comp, force_fit_comp, '-.', color=color, 
                 linewidth=1.5, alpha=0.8)
        
        # Withdrawal fit line - fixing to ensure it is plotted correctly
        if len(withdraw_data) > 0:
            # Find the range of withdrawal data that has meaningful force values
            high_force_wd = withdraw_data[withdraw_data['Force A'] > withdraw_data['Force A'].max() * 0.1]
            
            if len(high_force_wd) > 0:
                # Get the displacement range for the withdrawal phase (excluding very low force points)
                min_disp_wd = high_force_wd['trueD'].min()
                max_disp_wd = high_force_wd['trueD'].max()
                
                # Create fit line points
                disp_fit_wd = np.linspace(min_disp_wd, max_disp_wd, 10)
                
                # Use the correct formula: force = k_wd * (disp - disp_fin)
                # This will naturally create a line that eventually hits zero force at disp_fin
                force_fit_wd = results['k_wd'] * (disp_fit_wd - results['disp_fin'])
                
                # Plot only if we have valid points
                plt.plot(disp_fit_wd, force_fit_wd, ':', color=color, 
                         linewidth=1.5, alpha=0.8)
        
        # Mark yield point
        plt.scatter(results['disp_y'], results['f_y'], 
                    color=color, marker='o', s=50, edgecolors='black')
    
    # Format plot
    plt.ylim(-100, 1300)
    plt.xlabel('Displacement (µm)', fontsize=12)
    plt.ylabel('Force (µN)', fontsize=12)
    plt.title("Comparison of Force-Displacement Curves", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend for line styles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', lw=1.5, label='Compression'),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='Withdrawal'),
        Line2D([0], [0], color='gray', linestyle='-.', lw=1.5, label='k_ct fit'),
        Line2D([0], [0], color='gray', linestyle=':', lw=1.5, label='k_wd fit'),
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
    table_columns = ["Sample-Pillar", "k_ct (µN/µm)", "k_wd (µN/µm)", "k_mean (µN/µm)", "E_mean (MPa)", "F_y (µN)", "d_el (µm)", "d_pl (µm)"]
    
    for results, _ in all_results:
        k_avg = (results['k_wd'] + results['k_ct']) / 2
        E_avg = (results['E_wd'] + results['E_ct']) / 2
        row = [
            f"{results['sample']}-{results['pillar']}", 
            f"{results['k_ct']:.1f}",
            f"{results['k_wd']:.1f}",
            f"{k_avg:.1f}", 
            f"{E_avg:.1f}", 
            f"{results['f_y']:.1f}", 
            f"{results['disp_el']:.3f}", 
            f"{results['disp_pl']:.3f}"
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
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Make more room for the expanded table
    
    # Save figure if path provided
    if out_fig_path:
        plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_summary(sample, pillar, k_wd, k_ct, disp_fin, disp_el, disp_y, f_y, disp_pl):
    """Print a summary of the analysis results."""
    print(f"\nSample {sample}, pillar {pillar}:")
    print(f"Stiffness: withdrawal {k_wd:.1f} µN/µm, compression {k_ct:.1f} µN/µm, mean {(k_wd+k_ct)/2:.1f} µN/µm")
    print(f"Max plastic displacement, disp_fin:\t {disp_fin:.3f} µm")
    print(f"Max elastic displacement, disp_el:\t {disp_el:.3f} µm")
    print(f'Yield point:\t\t\t\t {disp_y:.3f} µm, {f_y:.1f} µN')
    print(f'Total plastic displacement:\t\t {disp_pl:.3f} µm')


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
            'k_ct (µN/µm)', 'k_wd (µN/µm)', 'k_avg (µN/µm)',
            'disp0 (µm)', 'disp_fin (µm)', 'disp_el (µm)', 'disp_pl (µm)', 
            'disp_y (µm)', 'F_y (µN)'
        ])
        
        # Write data for each sample-pillar
        for results, _ in all_results:
            writer.writerow([
                results['sample'],
                results['pillar'],
                results['test_n'],
                f"{results['k_ct']:.1f}",
                f"{results['k_wd']:.1f}",
                f"{(results['k_wd'] + results['k_ct'])/2:.1f}",
                f"{results['disp0']:.3f}",
                f"{results['disp_fin']:.3f}",
                f"{results['disp_el']:.3f}",
                f"{results['disp_pl']:.3f}",
                f"{results['disp_y']:.3f}",
                f"{results['f_y']:.1f}"
            ])
    
    print(f"Results saved to {csv_path}")


def batch_analyze_samples(json_path, sample_pillar_list, combined_fig_path=None, pillar_dimension_type="initial", test_id=0):
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
        results, data = analyze_force_displacement(
            json_path, sample, pillar, 
            pillar_dimension_type=pillar_dimension_type, 
            test_id=test_id,
            create_plot=False  # Don't create individual plots during batch analysis
        )
        all_results.append((results, data))
    
    # Create combined plot
    if combined_fig_path:
        create_combined_plot(all_results, combined_fig_path)
    
    return all_results


if __name__ == "__main__":
    # Example usage for multiple sample-pillars
    # json_path = r"/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DATA/FemtoTools/1st_series_pillar_CT_info.json"
    json_path = r"/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DATA/FemtoTools/CT_C_0001Zn_2025-02-26,10h43m44s/pillar_CT_info.json"
    
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
    
    combined_fig_path = outdir / "combined_force_displacement.png"
    csv_results_path = outdir / "all_sample_results_fd.csv"
    
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
    # results, data = analyze_force_displacement(json_path, '10-10', 'P1')