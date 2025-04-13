import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter


def find_inflection_points(epsilon, sigma, E, sampling_freq=500, visualise=False):
    """
    OLD METHOD USED PREVIOUSLY. CHOICE OF y_idx BY HAND.
    BILINEAR FIT CURRENTLY USED.
    Calculate the slope between consecutive data points and identify regions 
    where the slope remains relatively constant (indicating a linear trend). 
    You can then set thresholds for what constitutes a "relatively constant" slope.
    """
    smooth = gaussian_filter(sigma, 1)
    
    # Calculate derivatives
    epsilon_sampled = epsilon.iloc[::sampling_freq]
    smooth_sampled = smooth[::sampling_freq]
    d = np.diff(smooth_sampled) / np.diff(epsilon_sampled)  # First derivative
    dd = np.diff(d) / np.diff(epsilon_sampled.iloc[:-1])    # Second derivative
    
    # Threshold for detecting changes
    threshold = E * 1000        # 2024: 40000/5000 for 0001, 1000 for 11-20, 10000/ for 0001O
    change_indices = np.where(dd < -threshold)[0]
    
    # Convert indices back to original data indices
    infl_pts = change_indices * sampling_freq + epsilon.index[0]

    # yield_point_idx = infl_pts[1]                # choose yield point here
    # yield_strain = epsilon[yield_point_idx]
    # yield_stress = sigma[yield_point_idx]
    
    if visualise:
        plt.figure(figsize=(5, 3))
        plt.plot(d, alpha=0.3, label=f"d, n={sampling_freq}")
        plt.plot(dd,  alpha=0.3, label="dd")
        plt.scatter(change_indices, dd[change_indices], c='g', label='Flexion pts')
        plt.legend()
    #     plt.ylim([-1e9, 1e9])
    #     plt.xlim([0, 50])
        plt.show()

    return infl_pts


def bilinear_fit(data_x, data_y):
    """
    General function overview.
    """
    def error_function(breakpoint):
        idx = np.argmin(np.abs(data_x - breakpoint))
        
        # Fit first segment
        if idx > 2:  # Ensure enough points
            p1 = np.polyfit(data_x[:idx], data_y[:idx], 1)
            line1 = np.polyval(p1, data_x[:idx])
            err1 = np.sum((data_y[:idx] - line1)**2)
        else:
            return np.inf
            
        # Fit second segment
        if len(data_x) - idx > 2:
            p2 = np.polyfit(data_x[idx:], data_y[idx:] - (p1[0]*data_x[idx] + p1[1]) + (p1[0]*data_x[idx] + p1[1]), 1)
            line2 = np.polyval(p2, data_x[idx:])
            err2 = np.sum((data_y[idx:] - line2)**2)
        else:
            return np.inf
            
        return err1 + err2
    
    # Find optimal breakpoint
    result = minimize(lambda bp: error_function(bp[0]), 
                     [np.mean(data_x)], 
                     bounds=[(min(data_x), max(data_x))])
    
    return result.x[0]


def bilinear_fit_yield_point(strain, stress, min_points=10):
    """
    Find the yield point using a bi-linear fit method.
    
    This function identifies where a stress-strain curve transitions from elastic to plastic
    behavior by finding the optimal breakpoint that minimizes the error when fitting
    two separate linear segments.
    
    Parameters:
    -----------
    strain : pandas.Series
        Strain data
    stress : pandas.Series
        Stress data
    min_points : int, default=10
        Minimum number of points required for each segment
        
    Returns:
    --------
    tuple
        (yield_point_index, yield_strain, yield_stress, elastic_modulus)
    """
        
    # Ensure data is sorted by strain
    sorted_indices = np.argsort(strain.values)
    x = strain.values[sorted_indices]
    y = stress.values[sorted_indices]
    original_indices = strain.index.values[sorted_indices]
    
    # Make sure we have enough data points
    if len(x) < 2 * min_points:
        print(f"Warning: Not enough data points for bilinear fit. Need at least {2 * min_points}, have {len(x)}")
        # Return the highest stress point as fallback
        max_stress_idx = np.argmax(y)
        return original_indices[max_stress_idx], x[max_stress_idx], y[max_stress_idx], 0
    
    def error_function(breakpoint_idx):
        # Convert float to nearest index
        idx = int(np.clip(breakpoint_idx[0], min_points, len(x) - min_points))
        
        # Fit first segment (elastic region)
        p1 = np.polyfit(x[:idx], y[:idx], 1)
        line1 = np.polyval(p1, x[:idx])
        err1 = np.sum((y[:idx] - line1)**2)
        
        # Ensure continuity at breakpoint for second segment
        y_intercept = p1[0] * x[idx] + p1[1]
        
        # Fit second segment (plastic region) with a different slope
        if idx < len(x) - min_points:
            # Create x values for second segment
            x2 = x[idx:]
            
            # Create offset for second segment to ensure continuity
            p2 = np.polyfit(x2, y[idx:], 1)
            
            # Force the line to pass through the breakpoint
            p2_adjusted = [p2[0], y_intercept - p2[0] * x[idx]]
            
            line2 = np.polyval(p2_adjusted, x2)
            err2 = np.sum((y[idx:] - line2)**2)
        else:
            # If the breakpoint is too close to the end, penalize heavily
            err2 = np.inf
            
        # Add a small penalty for breakpoints near the edges
        edge_penalty = 0
        if idx < min_points * 2:
            edge_penalty = (min_points * 2 - idx) * np.mean(err1)
        elif idx > len(x) - min_points * 2:
            edge_penalty = (idx - (len(x) - min_points * 2)) * np.mean(err1)
            
        return err1 + err2 + edge_penalty
    
    # Initial guess: try several starting points and take the best
    best_error = np.inf
    best_start = len(x) // 2
    
    # Try different starting points to avoid local minima
    for start_point in [len(x) // 5, len(x) // 3, len(x) // 2, 2 * len(x) // 3]:
        result = minimize(
            error_function, 
            [start_point],
            bounds=[(min_points, len(x) - min_points)],
            method='L-BFGS-B'
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_start = result.x[0]
    
    # Get the optimal breakpoint
    optimal_idx = int(best_start)
    
    # Calculate elastic modulus from the first segment
    elastic_segment = np.polyfit(x[:optimal_idx], y[:optimal_idx], 1)
    elastic_modulus = elastic_segment[0]  # Slope of the elastic region
    
    # Return the yield point (index in original data, strain, stress, and E)
    yield_point_idx = original_indices[optimal_idx]
    yield_strain = x[optimal_idx]
    yield_stress = y[optimal_idx]
    
    return yield_point_idx, yield_strain, yield_stress, elastic_modulus


# Function to visualize the bilinear fit method results
def plot_bilinear_fit(strain, stress, yield_idx, ax=None):
    """
    Visualize the bilinear fit method by plotting both line segments.
    
    Parameters:
    -----------
    strain : pandas.Series
        Strain data
    stress : pandas.Series
        Stress data
    yield_idx : int
        Index of the yield point in the original data
    ax : matplotlib.axes, optional
        Axes to plot on. If None, a new figure is created.
    
    Returns:
    --------
    matplotlib.axes
        The axes with the plot
    """
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the yield point position
    yield_pos = strain.index.get_loc(yield_idx)
    
    # Sort data by strain for fitting
    sorted_indices = np.argsort(strain.values)
    x = strain.values[sorted_indices]
    y = stress.values[sorted_indices]
    
    # Find where the sorted index corresponds to our yield point
    for i, idx in enumerate(sorted_indices):
        if strain.index[idx] == yield_idx:
            sorted_yield_pos = i
            break
    
    # Fit the elastic region (before yield)
    p1 = np.polyfit(x[:sorted_yield_pos], y[:sorted_yield_pos], 1)
    x1_fit = np.linspace(x[0], x[sorted_yield_pos], 100)
    y1_fit = np.polyval(p1, x1_fit)
    
    # Calculate the continuation point for the plastic region
    y_intercept = p1[0] * x[sorted_yield_pos] + p1[1]
    
    # Fit the plastic region (after yield)
    p2 = np.polyfit(x[sorted_yield_pos:], y[sorted_yield_pos:], 1)
    
    # Force the line to pass through the yield point
    p2_adjusted = [p2[0], y_intercept - p2[0] * x[sorted_yield_pos]]
    
    x2_fit = np.linspace(x[sorted_yield_pos], x[-1], 100)
    y2_fit = np.polyval(p2_adjusted, x2_fit)
    
    # Plot original data
    ax.scatter(strain, stress, s=10, alpha=0.6, label='Data')
    
    # Plot the fits
    ax.plot(x1_fit, y1_fit, 'r-', linewidth=2, label='Elastic region fit')
    ax.plot(x2_fit, y2_fit, 'g-', linewidth=2, label='Plastic region fit')
    
    # Mark the yield point
    yield_strain = strain.iloc[yield_pos]
    yield_stress = stress.iloc[yield_pos]
    ax.scatter([yield_strain], [yield_stress], s=100, c='orange', 
               marker='o', edgecolors='black', label='Yield point')
    
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress (MPa)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax


def detect_first_plastic_event(strain, stress, window_size=5, threshold_factor=0.1, smoothing_factor=1):
    """
    Advanced method to detect the first plastic event in noisy micro-compression data.
    
    This function combines multiple approaches:
    1. Bi-linear fit to find the overall transition from elastic to plastic
    2. Local slope change detection for detecting discrete plastic events
    3. Adaptive noise filtering based on data characteristics
    
    Parameters:
    -----------
    strain : pandas.Series
        Strain data
    stress : pandas.Series
        Stress data
    window_size : int, default=5
        Window size for the rolling slope calculation
    threshold_factor : float, default=0.1
        Factor of elastic modulus to use as threshold for detecting slope changes
    smoothing_factor : float, default=1
        Factor for Gaussian smoothing (higher = more smoothing)
        
    Returns:
    --------
    tuple
        (yield_point_index, yield_strain, yield_stress, elastic_modulus)
    """
        
    # Step 1: First, get an estimate of the elastic region using bi-linear fit
    yield_idx, yield_strain, yield_stress, E_elastic = bilinear_fit_yield_point(strain, stress)
    
    # Get the position in the sorted data
    sorted_indices = np.argsort(strain.values)
    x = strain.values[sorted_indices]
    y = stress.values[sorted_indices]
    original_indices = strain.index.values[sorted_indices]
    
    # Find where the sorted index corresponds to the bilinear yield point
    for i, idx in enumerate(sorted_indices):
        if strain.index[idx] == yield_idx:
            bilinear_pos = i
            break
    
    # Step 2: Apply smoothing to the data
    # Use adaptive smoothing based on noise level
    noise_level = np.std(np.diff(y[:bilinear_pos//2])) / np.mean(y[:bilinear_pos//2])
    adaptive_smoothing = max(0.5, min(3.0, noise_level * 20 * smoothing_factor))
    
    print(f"Noise level: {noise_level:.4f}, Applied smoothing: {adaptive_smoothing:.2f}")
    
    y_smooth = gaussian_filter(y, adaptive_smoothing)
    
    # Step 3: Calculate slopes using rolling window
    slopes = []
    indices = []
    
    for i in range(window_size, len(x) - window_size):
        # Only look at points before the bilinear yield point + some margin
        if i > bilinear_pos * 1.5:
            break
            
        # Calculate local slope using surrounding points
        local_x = x[i-window_size:i+window_size]
        local_y = y_smooth[i-window_size:i+window_size]
        
        # Fit a line to get the slope
        p = np.polyfit(local_x, local_y, 1)
        slopes.append(p[0])
        indices.append(i)
    
    slopes = np.array(slopes)
    
    # Step 4: Detect significant slope changes
    # First, establish the baseline slope from the first part of the data
    baseline_idx = min(len(slopes) // 3, int(bilinear_pos * 0.5))
    baseline_slope = np.median(slopes[:baseline_idx])
    
    # Set threshold for significant deviation
    threshold = baseline_slope * (1 - threshold_factor)
    
    # Find where the slope drops significantly
    drops = np.where(slopes < threshold)[0]
    
    if len(drops) > 0:
        # Find the first significant drop that's sustained
        first_drop = drops[0]
        
        # Check if the drop is sustained (not just noise)
        sustained_drop = True
        check_range = min(5, len(slopes) - first_drop)
        
        if check_range > 2:
            # See if most points after the drop stay below threshold
            if np.mean(slopes[first_drop:first_drop+check_range] < threshold) < 0.6:
                sustained_drop = False
                
        if sustained_drop:
            event_idx = indices[first_drop]
            result_idx = original_indices[event_idx]
            result_strain = x[event_idx]
            result_stress = y[event_idx]
            
            print(f"First plastic event detected at index {result_idx}, strain {result_strain:.6f}, stress {result_stress:.2f} MPa")
            return result_idx, result_strain, result_stress, baseline_slope
    
    # If no clear plastic event is detected, fall back to the bilinear result
    print("No clear plastic event detected, using bilinear fit result instead")
    return yield_idx, yield_strain, yield_stress, E_elastic


def visualize_event_detection(strain, stress, event_idx, bilinear_idx=None, ax=None):
    """
    Visualize the first plastic event detection with slope changes.
    
    Parameters:
    -----------
    strain : pandas.Series
        Strain data
    stress : pandas.Series
        Stress data
    event_idx : int
        Index of the detected first plastic event
    bilinear_idx : int, optional
        Index of the yield point from bilinear method, for comparison
    ax : matplotlib.axes, optional
        Axes to plot on. If None, a new figure is created.
        
    Returns:
    --------
    matplotlib.axes
        The axes with the plot
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort data by strain
    sorted_indices = np.argsort(strain.values)
    x = strain.values[sorted_indices]
    y = stress.values[sorted_indices]
    original_indices = strain.index.values[sorted_indices]
    
    # Apply smoothing
    y_smooth = gaussian_filter(y, 1.0)
    
    # Plot original and smoothed data
    ax.plot(x, y, 'o-', color='lightgrey', markersize=2, alpha=0.7, label='Raw data')
    ax.plot(x, y_smooth, '-', color='blue', linewidth=1.5, label='Smoothed data')
    
    # Find the position of the event in the sorted data
    event_pos = np.where(original_indices == event_idx)[0][0]
    
    # Plot the event point
    ax.scatter([x[event_pos]], [y[event_pos]], s=150, c='red', 
               marker='o', edgecolors='black', label='First plastic event')
    
    # If bilinear yield point is provided, also plot it
    if bilinear_idx is not None:
        bilinear_pos = np.where(original_indices == bilinear_idx)[0][0]
        ax.scatter([x[bilinear_pos]], [y[bilinear_pos]], s=150, c='orange', 
                   marker='s', edgecolors='black', label='Bilinear yield point')
    
    # Calculate and plot slopes (on a second y-axis)
    window_size = 5
    slopes = []
    strain_points = []
    
    for i in range(window_size, len(x) - window_size):
        local_x = x[i-window_size:i+window_size]
        local_y = y_smooth[i-window_size:i+window_size]
        p = np.polyfit(local_x, local_y, 1)
        slopes.append(p[0])
        strain_points.append(x[i])
    
    # Create second y-axis for slopes
    ax2 = ax.twinx()
    ax2.plot(strain_points, slopes, '-', color='green', alpha=0.5, linewidth=1.5, label='Local slope')
    ax2.set_ylabel('Slope (MPa)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Mark the event on the slope curve
    if event_pos >= window_size and event_pos < len(x) - window_size:
        event_slope_idx = event_pos - window_size
        if 0 <= event_slope_idx < len(slopes):
            ax2.scatter([strain_points[event_slope_idx]], [slopes[event_slope_idx]], 
                       s=100, c='red', marker='^', edgecolors='black')
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress (MPa)')
    ax.grid(True, alpha=0.3)
    
    return ax, ax2

