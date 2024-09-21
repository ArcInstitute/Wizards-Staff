#!/usr/bin/env python
import os
import sys
import argparse

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
    parser.add_argument('metadata_path', type=str,
                        help='Path to the metadata file')
    parser.add_argument('--frate', type=int, default=30,
                        help='Frame rate of the video')
    parser.add_argument('--size-threshold', type=int, default=20000,
                        help='Parameter to filter out large noise') 
    parser.add_argument('--show-plots', action='store_true',
                        help='Set to True to show the plots inline')
    parser.add_argument('--save-files', action='store_true',
                        help='Set to True to save the output files')    
    parser.add_argument('--output-dir', type=str, default='lizard_wizard_outputs',
                        help='Directory to save the output files')
    parser.add_argument('--threads', type=int, default=2,
                        help='Number of parallel processes')
    return parser.parse_args()

## main interface function
def main():
    # parse cli args
    args = parse_args()

    # run the pipeline
    from wizards_staff.wizards.cauldron import run_all
    rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df = run_all(
        args.results_folder, 
        args.metadata_path,
        frate=args.frate, 
        size_threshold=args.size_threshold,
        show_plots=args.show_plots,
        save_files=args.save_files,
        output_dir=args.output_dir,
        threads=args.threads
    )

    # write out data tables
    os.makedirs(args.output_dir, exist_ok=True)
    rise_time_df.to_csv(os.path.join(args.output_dir, 'rise_time.csv'), index=False)
    fwhm_df.to_csv(os.path.join(args.output_dir, 'fwhm.csv'), index=False)
    frpm_df.to_csv(os.path.join(args.output_dir, 'frpm.csv'), index=False)
    mask_metrics_df.to_csv(os.path.join(args.output_dir, 'mask_metrics.csv'), index=False)
    silhouette_scores_df.to_csv(os.path.join(args.output_dir, 'silhouette_scores.csv'), index=False)

## script main
if __name__ == '__main__': 
    main()