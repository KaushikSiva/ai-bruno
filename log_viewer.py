#!/usr/bin/env python3
"""
Bruno GPT Vision Log Viewer
Real-time log monitoring for Bruno robot behavior
"""

import time
import os
import sys
from datetime import datetime
import argparse

def follow_log_file(log_file="bruno_gpt.log", lines=10):
    """Follow a log file in real-time"""
    print(f"Following log file: {log_file}")
    print("=" * 80)
    
    # Show last few lines if file exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines_list = f.readlines()
            if lines_list:
                print("Last few log entries:")
                for line in lines_list[-lines:]:
                    print(line.rstrip())
                print("-" * 80)
    
    # Follow the file
    try:
        with open(log_file, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    # Color code different log levels
                    if "ERROR" in line:
                        print(f"\033[91m{line.rstrip()}\033[0m")  # Red
                    elif "WARNING" in line:
                        print(f"\033[93m{line.rstrip()}\033[0m")  # Yellow
                    elif "INFO" in line:
                        print(f"\033[94m{line.rstrip()}\033[0m")  # Blue
                    elif "DEBUG" in line:
                        print(f"\033[90m{line.rstrip()}\033[0m")  # Gray
                    else:
                        print(line.rstrip())
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping log viewer...")
    except FileNotFoundError:
        print(f"Log file {log_file} not found. Start Bruno first.")

def show_log_summary(log_file="bruno_gpt.log"):
    """Show a summary of the log file"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("Log file is empty.")
        return
    
    print(f"Log Summary for {log_file}")
    print("=" * 50)
    print(f"Total lines: {len(lines)}")
    
    # Count log levels
    error_count = sum(1 for line in lines if "ERROR" in line)
    warning_count = sum(1 for line in lines if "WARNING" in line)
    info_count = sum(1 for line in lines if "INFO" in line)
    debug_count = sum(1 for line in lines if "DEBUG" in line)
    
    print(f"ERROR: {error_count}")
    print(f"WARNING: {warning_count}")
    print(f"INFO: {info_count}")
    print(f"DEBUG: {debug_count}")
    
    # Find key events
    print("\nKey Events:")
    for line in lines:
        if any(keyword in line for keyword in ["STATE CHANGE", "BOTTLE DETECTED", "BIN DETECTED", "EMERGENCY", "TASK COMPLETE"]):
            print(f"  {line.strip()}")

def filter_log(log_file="bruno_gpt.log", level=None, keyword=None):
    """Filter log entries"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    filtered_lines = []
    for line in lines:
        include = True
        
        if level and level.upper() not in line:
            include = False
        
        if keyword and keyword.lower() not in line.lower():
            include = False
        
        if include:
            filtered_lines.append(line)
    
    print(f"Filtered log entries ({len(filtered_lines)} of {len(lines)}):")
    print("=" * 50)
    for line in filtered_lines:
        print(line.rstrip())

def main():
    parser = argparse.ArgumentParser(description="Bruno GPT Vision Log Viewer")
    parser.add_argument("--log-file", default="bruno_gpt.log", help="Log file to monitor")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log file in real-time")
    parser.add_argument("--summary", "-s", action="store_true", help="Show log summary")
    parser.add_argument("--filter", help="Filter by keyword")
    parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Filter by log level")
    parser.add_argument("--lines", type=int, default=10, help="Number of lines to show initially")
    
    args = parser.parse_args()
    
    if args.follow:
        follow_log_file(args.log_file, args.lines)
    elif args.summary:
        show_log_summary(args.log_file)
    elif args.filter or args.level:
        filter_log(args.log_file, args.level, args.filter)
    else:
        # Default: show last few lines
        if os.path.exists(args.log_file):
            with open(args.log_file, 'r') as f:
                lines = f.readlines()
                print(f"Last {args.lines} lines of {args.log_file}:")
                print("=" * 50)
                for line in lines[-args.lines:]:
                    print(line.rstrip())
        else:
            print(f"Log file {args.log_file} not found.")
            print("Start Bruno with: python3 gpt.py")

if __name__ == "__main__":
    main()
