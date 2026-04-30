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
    parser.add_argument('--allow-missing', action='store_true', default=False,
                        help='Allow missing input files')
    parser.add_argument('--remove-outlier', action='store_true', default=False,
                        help=('Exclude detected outlier neurons from downstream '
                              'metric DataFrames and per-shard plots, and emit '
                              'population-mean / per-neuron event plots.'))
    parser.add_argument('--show-outlier', action='store_true', default=False,
                        help=('When combined with --remove-outlier, also emit '
                              '"with outliers" copies of every affected plot and '
                              'CSV under <output-dir>/with_outliers/. No effect '
                              'without --remove-outlier.'))
    parser.add_argument('--filter-events', action='store_true', default=False,
                        help=('Master switch for per-event filtering. When set, '
                              'the --min-event-* / --max-event-* bounds below '
                              'are applied to drop noise / artifact events. '
                              'When unset (default), all detected events are '
                              'kept regardless of the bounds.'))
    parser.add_argument('--min-event-amplitude', type=float, default=0.05,
                        help=('Per-event filter (requires --filter-events): '
                              'drop events whose peak dF/F is below this value. '
                              'Default 0.05 rejects sub-noise and negative '
                              '"peaks". Set to a negative number to effectively '
                              'disable the lower bound.'))
    parser.add_argument('--max-event-amplitude', type=float, default=10.0,
                        help=('Per-event filter (requires --filter-events): '
                              'drop events whose peak dF/F exceeds this value. '
                              'Default 10.0 rejects deconvolution numerical '
                              'artifacts.'))
    parser.add_argument('--min-event-fwhm', type=int, default=2,
                        help=('Per-event filter (requires --filter-events): '
                              'drop events whose FWHM (in frames) is below '
                              'this value. Default 2 rejects 1-frame "spikes".'))
    parser.add_argument('--max-event-fwhm', type=int, default=None,
                        help=('Per-event filter (requires --filter-events): '
                              'drop events whose FWHM (in frames) exceeds this '
                              'value. Default None (no upper bound).'))
    parser.add_argument('--indicator', type=str, default=None,
                        help=('Calcium indicator preset for the waveform '
                              'outlier detector. Loads published-kinetics '
                              'rise/decay/peak-height defaults from '
                              'INDICATOR_PRESETS. Examples: GCaMP6f, '
                              'GCaMP6s, GCaMP6m, GCaMP7f, jGCaMP8f, '
                              'jGCaMP8m, jGCaMP8s, jRGECO1a, jRCaMP1a, '
                              'GCaMP3. Required when working with anything '
                              'other than GCaMP6f-like green indicators; '
                              'mismatched templates silently flag real '
                              'events as shape outliers. Default None '
                              '(legacy GCaMP6f-like template).'))
    parser.add_argument('--template-rise-ms', type=float, default=None,
                        help=('Override the waveform template rise time '
                              '(ms). Wins over --indicator preset. Default '
                              'None (use preset / GCaMP6f-like 50 ms).'))
    parser.add_argument('--template-decay-ms', type=float, default=None,
                        help=('Override the waveform template decay '
                              'time-constant (ms). Wins over --indicator '
                              'preset. Default None (use preset / '
                              'GCaMP6f-like 400 ms).'))
    parser.add_argument('--template-total-ms', type=float, default=None,
                        help=('Override the waveform template total '
                              'length (ms). Default None (1500 ms).'))
    parser.add_argument('--peak-height', type=float, default=None,
                        help=('Override the absolute dF/F peak threshold '
                              'used by the waveform detector. Wins over '
                              '--indicator preset. Default None (use '
                              'preset / GCaMP6f-like 0.10). Red '
                              'indicators usually need ~0.05.'))
    parser.add_argument('--no-report', dest='generate_report',
                        action='store_false', default=True,
                        help=('Silence the end-of-run summary report. By '
                              'default a markdown report is printed (and '
                              'written to <output-dir>/run_report.md when '
                              'output files are saved).'))
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
    orb = Orb(args.results_folder, args.metadata_path, allow_missing=args.allow_missing)

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
        zscore_threshold=3,
        remove_outlier=args.remove_outlier,
        show_outlier=args.show_outlier,
        filter_events=args.filter_events,
        min_event_amplitude=args.min_event_amplitude,
        max_event_amplitude=args.max_event_amplitude,
        min_event_fwhm=args.min_event_fwhm,
        max_event_fwhm=args.max_event_fwhm,
        indicator=args.indicator,
        template_rise_ms=args.template_rise_ms,
        template_decay_ms=args.template_decay_ms,
        template_total_ms=args.template_total_ms,
        peak_height=args.peak_height,
        generate_report=args.generate_report,
    )

    # Status
    print(f"Output written to: {args.output_dir}")


## script main
if __name__ == '__main__':  
    main(args)