#!/usr/bin/env python
# import
## batteries
import os
import sys
import argparse
## import from package
from wizards_staff.wizards.orb import Orb

# argparse
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

def parse_args():
    desc = 'Wizards Staff CLI'
    epi = """DESCRIPTION:
    Process the results folder, compute metrics, and write out data tables.
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                     formatter_class=CustomFormatter)
    parser.add_argument('results_folder', type=str,
                        help='Lizard-Wizard pipeline results folder')
    parser.add_argument('--metadata-path', type=str, default=None,
                        help='Path to the metadata file. If not provided, then assumed to be {results_folder}/metadata.csv')
    parser.add_argument('--group-name', type=str, default='Well',
                        help='Group name in the metadata file')
    parser.add_argument('--frate', type=int, default=30,
                        help='Frame rate of the video')
    parser.add_argument('--size-threshold', type=int, default=20000,
                        help='Parameter to filter out large noise') 
    parser.add_argument('--percent-threshold', type=float, default=0.2,
                        help='Percentage threshold for FWHM calculation')
    parser.add_argument('--p-th', type=float, default=75,
                        help='Percentile threshold for image processing')
    parser.add_argument('--zscore-threshold', type=int, default=3,
                        help='Z-score threshold for filtering out noise events')
    parser.add_argument('--min-clusters', type=int, default=2,
                        help='The minimum number of clusters to try')
    parser.add_argument('--max-clusters', type=int, default=10,
                        help='The maximum number of clusters to try')
    parser.add_argument('--output-dir', type=str, default='wizards-staff_output',
                        help='Directory to save the output files')
    parser.add_argument('--threads', type=int, default=2,
                        help='Number of parallel processes')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode')
    return parser.parse_args()

## main interface function
def main():
    # parse args
    args = parse_args()

    # metadata path
    if args.metadata_path is None:
        print(f'No metadata path provided. Assuming {args.results_folder}/metadata.csv', file=sys.stderr)
        args.metadata_path = os.path.join(args.results_folder, 'metadata.csv')

    # check input
    if not os.path.exists(args.metadata_path):
        sys.exit(f'Metadata file does not exist: {args.metadata_path}')
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.results_folder):
        sys.exit(f'Results folder does not exist: {args.results_folder}')

    # run the full pipeline
    orb = Orb(args.results_folder, args.metadata_path)

    # run all
    orb.run_all(
        group_name=args.group_name, 
        show_plots=False, 
        save_files=True, 
        output_dir=args.output_dir, 
        threads=args.threads, 
        debug=args.debug,
        frate=args.frate,
        size_threshold=args.size_threshold,
        percentage_threshold=0.2,
        p_th=75,
        min_clusters=2,
        max_clusters=10,
        zscore_threshold=3
    )

    # Status
    print(f"Output written to: {args.output_dir}")


## script main
if __name__ == '__main__':  
    main(args)