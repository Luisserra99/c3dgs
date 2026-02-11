import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_metrics(metrics_file: str = "extracted_metrics.json") -> Dict:
    """Load the extracted metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def extract_directory_and_subdir(label: str) -> tuple:
    """
    Extract directory name and subdirectory number from label.
    Returns (directory_name, subdir_number)
    """
    parts = label.rsplit('_', 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return label, 0
    return label, 0


def generate_graphs(metrics_file: str = "extracted_metrics.json", output_dir: str = "metric_graphs") -> None:
    """
    Generate 4 graphs (one for each metric: SSIM, PSNR, LPIPS, size).
    
    Args:
        metrics_file: Path to the extracted metrics JSON file
        output_dir: Directory to save the generated graphs
    """
    
    # Load metrics
    metrics_data = load_metrics(metrics_file)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Metrics to plot
    metric_names = ["SSIM", "PSNR", "LPIPS", "size"]
    
    # Create a figure for each metric
    for metric_name in metric_names:
        metric_values = metrics_data[metric_name]
        
        if not metric_values:
            print(f"No data for metric: {metric_name}")
            continue
        
        # Organize data by directory and subdirectory
        directories = {}
        for label, value in metric_values.items():
            dir_name, subdir_num = extract_directory_and_subdir(label)
            if dir_name not in directories:
                directories[dir_name] = {}
            directories[dir_name][subdir_num] = value
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Prepare data for plotting
        x_labels = [f"{i}" for i in range(1, 14)]
        x_pos = np.arange(len(x_labels))
        width = 0.15  # Width of bars for each directory
        
        # Plot bars for each directory
        dir_names = sorted(directories.keys())
        for idx, dir_name in enumerate(dir_names):
            values = [directories[dir_name].get(i, 0) for i in range(1, 14)]
            offset = (idx - len(dir_names) / 2 + 0.5) * width
            ax.bar(x_pos + offset, values, width, label=dir_name, alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel("Subdirectory Number", fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f"{metric_name} Comparison Across Directories and Subdirectories", 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend(title="Directory", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Save the figure
        output_path = Path(output_dir) / f"{metric_name.lower()}_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph saved: {output_path}")
    
    print(f"\nAll graphs saved to {output_dir}")


def generate_line_graphs(metrics_file: str = "extracted_metrics.json", output_dir: str = "metric_graphs") -> None:
    """
    Alternative: Generate line graphs (one line per directory).
    
    Args:
        metrics_file: Path to the extracted metrics JSON file
        output_dir: Directory to save the generated graphs
    """
    
    # Load metrics
    metrics_data = load_metrics(metrics_file)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Metrics to plot
    metric_names = ["SSIM", "PSNR", "LPIPS", "size"]
    
    # Create a figure for each metric
    for metric_name in metric_names:
        metric_values = metrics_data[metric_name]
        
        if not metric_values:
            print(f"No data for metric: {metric_name}")
            continue
        
        # Organize data by directory and subdirectory
        directories = {}
        for label, value in metric_values.items():
            dir_name, subdir_num = extract_directory_and_subdir(label)
            if dir_name not in directories:
                directories[dir_name] = {}
            directories[dir_name][subdir_num] = value
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot line for each directory
        dir_names = sorted(directories.keys())
        for dir_name in dir_names:
            x_positions = sorted(directories[dir_name].keys())
            values = [directories[dir_name][x] for x in x_positions]
            ax.plot(x_positions, values, marker='o', label=dir_name, linewidth=2, markersize=6)
        
        # Customize the plot
        ax.set_xlabel("Subdirectory Number", fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f"{metric_name} Across Subdirectories", fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 14))
        ax.legend(title="Directory", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        output_path = Path(output_dir) / f"{metric_name.lower()}_line.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph saved: {output_path}")
    
    print(f"\nAll line graphs saved to {output_dir}")


if __name__ == "__main__":
    # Generate bar graphs
    print("Generating bar graphs...")
    generate_graphs()
    
    # Optionally, also generate line graphs
    print("\nGenerating line graphs...")
    generate_line_graphs()
