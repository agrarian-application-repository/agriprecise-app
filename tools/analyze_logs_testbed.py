import re
import argparse
import sys

def analyze_log(file_paths):
    """Summarise RTK fix rates across one or more server logs.

    Several files are accepted because log rotation splits the history between
    server_execution.log and its server_execution.log.<date> siblings.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    total_cycles = 0
    fixed_cycles = 0
    
    total_epochs_processed = 0
    total_fixed_epochs = 0
    
    # Initialize extreme values
    best_batch = {"pct": -1.0, "fixed": 0, "total": 0}
    worst_batch = {"pct": 101.0, "fixed": 0, "total": 0}
    
    max_total_epochs = 0
    min_total_epochs = float('inf')
    
    percentages = []

    # Regular expression to extract data from the specific log line format
    # Example: "RTK Result: 158/180 FIXED epochs (87.8%)"
    rtk_pattern = re.compile(r"RTK Result:\s+(\d+)/(\d+)\s+FIXED epochs\s+\(([\d\.]+)\%\)")

    files_read = 0
    for file_path in file_paths:
        try:
            file = open(file_path, 'r')
        except FileNotFoundError:
            print(f"Warning: File '{file_path}' not found, skipping.")
            continue

        files_read += 1
        with file:
            for line in file:
                # Count every time the system attempts to process a batch
                if "Window closed" in line:
                    total_cycles += 1
                
                match = rtk_pattern.search(line)
                if match:
                    fixed_epochs = int(match.group(1))
                    total_epochs = int(match.group(2))
                    pct = float(match.group(3))
                    
                    # Accumulate global metrics
                    fixed_cycles += 1
                    total_epochs_processed += total_epochs
                    total_fixed_epochs += fixed_epochs
                    percentages.append(pct)
                    
                    # Track Min/Max total epochs per batch
                    if total_epochs > max_total_epochs:
                        max_total_epochs = total_epochs
                    if total_epochs < min_total_epochs:
                        min_total_epochs = total_epochs
                    
                    # Track Best performance batch
                    if pct > best_batch["pct"]:
                        best_batch = {"pct": pct, "fixed": fixed_epochs, "total": total_epochs}
                        
                    # Track Worst performance batch
                    if pct < worst_batch["pct"]:
                        worst_batch = {"pct": pct, "fixed": fixed_epochs, "total": total_epochs}

    if files_read == 0:
        print("Error: could not read any log file.")
        sys.exit(1)

    if fixed_cycles == 0:
        print("No FIXED results found in the log.")
        sys.exit(0)

    # Calculate final averages
    avg_pct = sum(percentages) / len(percentages)
    overall_ratio = (total_fixed_epochs / total_epochs_processed) * 100 if total_epochs_processed > 0 else 0

    # Print the report
    print("-" * 55)
    print("RTK PROCESSING LOG ANALYSIS REPORT")
    print("-" * 55)
    print("GENERAL STATISTICS:")
    print(f"Total processing cycles attempted: {total_cycles}")
    print(f"Cycles achieving FIXED status:     {fixed_cycles}")
    print(f"Total epochs processed across log: {total_epochs_processed}")
    print(f"Total FIXED epochs achieved:       {total_fixed_epochs}")
    print(f"Overall data FIXED ratio:          {overall_ratio:.2f}%")
    print(f"Average FIXED percentage per file: {avg_pct:.2f}%")
    print("-" * 55)
    print("BATCH PERFORMANCE (BEST & WORST):")
    print(f"Best calculated batch:             {best_batch['pct']:.1f}% ({best_batch['fixed']}/{best_batch['total']} epochs)")
    print(f"Worst calculated batch:            {worst_batch['pct']:.1f}% ({worst_batch['fixed']}/{worst_batch['total']} epochs)")
    print("-" * 55)
    print("EPOCH DISTRIBUTION:")
    print(f"Maximum epochs in a single batch:  {max_total_epochs}")
    print(f"Minimum epochs in a single batch:  {min_total_epochs}")
    print("-" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse server logs and generate RTK statistics.")
    parser.add_argument("log_file", type=str, nargs='+',
                        help="Log file(s). With rotation enabled: logs/server_execution.log*")

    args = parser.parse_args()
    analyze_log(args.log_file)