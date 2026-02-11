import json
import os
from pathlib import Path
from typing import List, Dict

def extract_metrics(directories: List[str], output_file: str = "extracted_metrics.json") -> None:
    """
    Extract metrics from results.json files in subdirectories.
    
    Args:
        directories: List of root directories to process
        output_file: Path to save the compiled metrics
    """
    
    # Subdirectories mapping (number -> name)
    subdir_names = [
        "bicycle", "bonsai", "counter", "drjohnson", "flowers", 
        "garden", "kitchen", "playroom", "room", "stump", 
        "train", "treehill", "truck"
    ]
    
    # Initialize structure for metrics
    metrics_data = {
        "SSIM": {},
        "PSNR": {},
        "LPIPS": {},
        "size": {}
    }
    
    # Process each directory
    for dir_name in directories:
        dir_path = Path(dir_name)
        
        if not dir_path.exists():
            print(f"Warning: Directory {dir_name} does not exist")
            continue
        
        print(f"Processing directory: {dir_name}")
        
        # Process each subdirectory
        for subdir_num, subdir_name in enumerate(subdir_names, start=1):
            results_file = dir_path / subdir_name / "results.json"
            
            if not results_file.exists():
                print(f"  Warning: {results_file} not found")
                continue
            
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                # Extract metrics from the first key (e.g., "ours_35000")
                key = list(data.keys())[0]
                metrics = data[key]
                
                # Store metrics by subdirectory number
                metrics_data["SSIM"][f"{dir_name}_{subdir_num}"] = metrics["SSIM"]
                metrics_data["PSNR"][f"{dir_name}_{subdir_num}"] = metrics["PSNR"]
                metrics_data["LPIPS"][f"{dir_name}_{subdir_num}"] = metrics["LPIPS"]
                metrics_data["size"][f"{dir_name}_{subdir_num}"] = metrics["size"]
                
                print(f"  Subdir {subdir_num} ({subdir_name}): OK")
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"  Error processing {results_file}: {e}")
    
    # Save the compiled metrics
    with open(output_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\nMetrics saved to {output_file}")
    return metrics_data


if __name__ == "__main__":
    # Example usage
    # Modify this list with your actual directory paths
    directories = [
        "/d01/luis/outputs_preliminares",
        "/d01/luis/outputs_preliminares_hilbert",
        "/d01/luis/outputs_preliminares_hilbert_18",
        "/d01/luis/outputs_preliminares_sensitivity_softmax",
    ]
    
    extract_metrics(directories)
