#!/usr/bin/env python3
"""
run_analysis.py

Wrapper script to run comprehensive analysis and save results to data-log/output/{timestamp}
with metadata about which videos were analyzed.

Usage:
    python3 run_analysis.py [--fine-data-dir ./fine_data] [--output-dir data-log/output]
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add data-log to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data-log'))

from analysis_viz import run_analysis


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive analysis on danger dataset')
    parser.add_argument('--fine-data-dir', default='./fine_data', help='Path to fine_data directory')
    parser.add_argument('--output-dir', default='./data-log/output', help='Output directory base path')
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'[Analysis Wrapper]')
    print(f'  Fine data directory: {args.fine_data_dir}')
    print(f'  Output directory: {args.output_dir}')
    
    # Run analysis
    result_dir = run_analysis(args.fine_data_dir, output_subdir=None)
    
    if result_dir:
        print(f'\n✓ Analysis complete')
        print(f'  Results saved to: {result_dir}')
        print(f'  Files:')
        for fname in sorted(os.listdir(result_dir)):
            fpath = os.path.join(result_dir, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                size_str = f'{size/1024:.1f}KB' if size > 1024 else f'{size}B'
                print(f'    - {fname} ({size_str})')
        
        return 0
    else:
        print(f'\n✗ Analysis failed')
        return 1


if __name__ == '__main__':
    sys.exit(main())
