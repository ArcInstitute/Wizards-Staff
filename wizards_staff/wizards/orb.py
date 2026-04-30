# import
## batteries
import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from functools import wraps
## 3rd party
import numpy as np
import pandas as pd
from tifffile import imread
from tqdm.notebook import tqdm
## package
from wizards_staff.logger import init_custom_logger
from wizards_staff.pwc import run_pwc as ws_run_pwc
from wizards_staff.wizards.shard import Shard
from wizards_staff.wizards.cauldron import run_all as ws_run_all

# Functions
def npy_loader(infile, allow_pickle=True):
    return np.load(infile, allow_pickle=allow_pickle)

# Data item mapping (how to load each data item)
DATA_ITEM_MAPPING = {
    'cnm_A': {
        'suffixes': ['_cnm-A.npy'],
        'loader': npy_loader,
        'required': True
    },
    'cnm_C': {
        'suffixes': ['_cnm-C.npy'],
        'loader': npy_loader,
        'required': True
    },
    'cnm_S': {
        'suffixes': ['_cnm-S.npy'],
        'loader': npy_loader,
        'required': True
    },
    'cnm_idx': {
        'suffixes': ['_cnm-idx.npy'],
        'loader': npy_loader,
        'required': True
    },
    'df_f0_graph': {
        'suffixes': ['_df-f0-graph.tif'],
        'loader': imread,
        'required': False
    },
    'dff_dat': { 
        'suffixes': ['_dff-dat.npy'],
        'loader': npy_loader,
        'required': True
    },
    'f_dat': { 
        'suffixes': ['_f-dat.npy'],
        'loader': npy_loader,
        'required': True
    },
    'minprojection': {    # aka: im_min
        'suffixes': ['_minprojection.tif'],
        'loader': imread,
        'required': True
    },
    'mask': {  
        'suffixes': ['_masks.tif'],
        'loader': imread,
        'required': True
    }
}

# classes
@dataclass
class Orb:
    """
    Represents a collection of samples and their associated data shards.
    """
    results_folder: str
    metadata_file_path: str
    metadata: pd.DataFrame = field(init=False)
    allow_missing: bool = False
    quiet: bool = False
    _logger: Optional[logging.Logger] = field(default=None, init=False)
    _rise_time_data: pd.DataFrame = field(default=None, init=False)
    _fall_time_data: pd.DataFrame = field(default=None, init=False)
    _fwhm_data: pd.DataFrame = field(default=None, init=False)
    _frpm_data: pd.DataFrame = field(default=None, init=False)
    _peak_amplitude_data: pd.DataFrame = field(default=None, init=False)
    _max_peak_amplitude_data: pd.DataFrame = field(default=None, init=False)
    _peak_to_peak_data: pd.DataFrame = field(default=None, init=False)
    _mask_metrics_data: pd.DataFrame = field(default=None, init=False)
    _silhouette_scores_data: pd.DataFrame = field(default=None, init=False)
    _outlier_data: pd.DataFrame = field(default=None, init=False)
    _event_drop_log: pd.DataFrame = field(default=None, init=False)
    _shards: Dict[str, Shard] = field(default_factory=dict, init=False)   # loaded data
    _data_mapping: Dict[str, Any] = field(default_factory=lambda: DATA_ITEM_MAPPING, init=False)  # data item mapping
    _input_files: pd.DataFrame = field(default=None, init=False)  # file paths
    _input: pd.DataFrame = field(default=None, init=False)  # all input data
    _samples: set = field(default=None, init=False)  # samples
    _df_mn_pwc: pd.DataFrame = field(default=None, init=False)  # pairwise correlation results
    _df_mn_pwc_intra: pd.DataFrame = field(default=None, init=False)  # pairwise correlation results
    _df_mn_pwc_inter: pd.DataFrame = field(default=None, init=False)  # pairwise correlation results
    _pwc_plots: Dict[str, Any] = field(default_factory=dict, init=False)  # pairwise correlation results
    # Outlier-handling mode set by run_all(); accessors filter by these flags.
    _remove_outlier: bool = field(default=False, init=False)
    _show_outlier: bool = field(default=False, init=False)
    
    def __post_init__(self):
        # Configure logging
        self._logger = init_custom_logger(__name__)
        # load metadata
        self._load_metadata(self.metadata_file_path)
        self._samples = set(self.metadata['Sample'])
        # run categorization upon initialization
        self._categorize_files()   

    def _categorize_files(self):
        """
        Categorizes files into corresponding data items for each sample.
        """
        self._logger.info("Categorizing files...")
        mask_suffixes = {}
        # load files 
        for file_path in self._list_files(self.results_folder):
            # skip concatenated/output/ folder
            file_path_parts = file_path.split(os.path.sep)
            try:
                if file_path_parts[-3] == "concatenated" and file_path_parts[-2] == "output":
                    continue
            except IndexError:
                pass
            # get file info 
            ## basename
            file_basename = os.path.basename(file_path)
            ## sample name
            sample_name = None
            for sample in self.metadata['Sample']:
                if file_basename.startswith(sample):
                    sample_name = sample
                    break
            ### filter out samples not in metadata
            if sample_name is None or sample_name not in self._samples:
                continue
            ## suffix
            file_suffix = file_basename[len(sample_name):]

            # check for mask suffix consistency 
            # ie make sure we arent pulling both masked and unmasked files
            mask_part = file_suffix.split('_')[0]
            if sample_name in mask_suffixes:
                assert mask_suffixes[sample_name] == mask_part, (
                    f"Inconsistent mask suffix for sample {sample_name}: "
                    f"expected '{mask_suffixes[sample_name]}', got '{mask_part}'"
                )
            else:
                mask_suffixes[sample_name] = mask_part
        
            ## categorize file based on suffix
            for item_name, data_info in self._data_mapping.items():
                # file suffix matches data item suffix?
                if any(file_suffix.endswith(suffix) for suffix in data_info['suffixes']):
                    shard = self._shards.setdefault(
                        sample_name,
                        Shard(
                            sample_name,
                            metadata=self.metadata[self.metadata['Sample'] == sample_name],
                            files={},
                            quiet=self.quiet,
                            allow_missing=self.allow_missing
                        )
                    )
                    if item_name in shard.files:
                        existing_file, _ = shard.files[item_name]
                        # Extract the previously recorded suffix
                        existing_suffix = existing_file.split(sample_name, 1)[-1]
                        # Assert that the new suffix is identical to the existing one
                        assert existing_suffix == file_suffix, (
                            f"Inconsistent suffix for sample {sample_name} and data item '{item_name}': "
                            f"'{existing_suffix}' vs '{file_suffix}'"
                        )
                    shard.files[item_name] = (file_path, data_info['loader'])
                    break
        # check for missing files
        for item_name, item_info in self._data_mapping.items():
            if not item_info.get('required', True):  # Skip warning if item is not required
                continue
            missing_samples = []
            for sample in self._samples:
                try:
                    # sample has file?
                    if self._shards[sample].has_file(item_name) is False:
                        missing_samples.append(sample)
                except KeyError:
                    # no sample?
                    missing_samples.append(sample)
            if len(missing_samples) > 0 and not self.quiet:
                missing_samples = ', '.join(missing_samples)
                msg = f"WARNING: No '{item_name}' files found for samples: {missing_samples}"
                print(msg, file=sys.stderr)
    
    def _load_metadata(self, metadata_file_path: str):
        """
        Loads metadata from a CSV file.
        Args:
            metadata_file_path: Path to the metadata CSV file.
        """
        # status
        self._logger.info(f"Loading metadata from: {metadata_file_path}")
        # check if metadata file exists
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
        # load metadata
        self.metadata = pd.read_csv(metadata_file_path)
        # check for required columns
        required_columns = {"Sample", "Well", "Frate"}
        missing_columns = required_columns - set(self.metadata.columns)
        if missing_columns:
            cols_str = ', '.join(missing_columns)
            raise ValueError(f"Missing columns in metadata file: {cols_str}")

    #-- get items --#
    def shatter(self) -> Generator[Shard, None, None]:
        """
        Yields each Shard (sample's data) in the Orb.
        """
        yield from self._shards.values()

    def items(self) -> Generator[Tuple[str, Shard], None, None]:
        """
        Yields tuples of sample names and their Shard objects.
        """
        yield from self._shards.items()

    def _get_shard_data(self, attr_name: str) -> pd.DataFrame:
        """
        Dynamically generate a DataFrame for the given attribute from shards.
        """
        attr = getattr(self, attr_name)
        if attr is None:
            # Create DataFrame if it doesn't exist
            DF = []
            for shard in self.shatter():
                shard_data = getattr(shard, attr_name, None)
                if shard_data is not None:
                    DF += shard_data
            if len(DF) == 0:
                return None
            attr = pd.DataFrame(DF)
            # Cache the result
            setattr(self, attr_name, attr)  
        return attr

    def _filter_outliers(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Filter rows where ``is_outlier=True`` when ``self._remove_outlier`` is set
        and the column is present. Returns ``df`` unchanged otherwise.
        """
        if df is None:
            return None
        if self._remove_outlier and 'is_outlier' in df.columns:
            df = df[~df['is_outlier'].astype(bool)]
        return df

    #-- check input --#
    def has_input(self, data_item_name: str) -> bool:
        """
        Checks if the input data item exists.
        Args:
            data_item_name: Name of the data item.
        Returns:
            True if the input data item exists.
        """
        return any(self.input_files["DataItem"] == data_item_name)

    #-- save data --#
    def save_results(self, outdir: str, result_names: 
                  list=["rise_time_data", "fall_time_data", "fwhm_data", "frpm_data",
                        "peak_amplitude_data", "max_peak_amplitude_data",
                        "well_peak_amplitude_data",
                        "peak_to_peak_data",
                        "mask_metrics_data", "silhouette_scores_data",
                        "outlier_data", "event_drop_log",
                        "df_mn_pwc", "df_mn_pwc_intra", "df_mn_pwc_inter"]):
        """
        Saves data items to disk.

        When ``run_all`` was called with ``show_outlier=True``, the metric
        DataFrames that have a ``*_with_outliers`` companion are also written
        (full, un-cleaned views) into ``<outdir>/with_outliers/``.

        Args:
            outdir: Output directory.
            result_names: List of results to save.
        """
        self._logger.info(f"Saving data items to: {outdir}")
        # output directory
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # save each data item
        outfiles = []
        for name in result_names:
            # get property
            data = getattr(self, name)
            if data is None:
                self._logger.warning(f"Data item '{name}' not found")
                continue
            # write to disk. The drop ledger keeps its underscore name
            # (matches the in-memory schema and the spec'd file layout)
            # rather than being dash-cased like the other metric CSVs,
            # so downstream tooling can resolve it by a stable filename.
            if name == "event_drop_log":
                outfile = os.path.join(outdir, "event_drop_log.csv")
            else:
                outfile = os.path.join(outdir, name.replace("_", "-") + ".csv")
            data.to_csv(outfile, index=False)
            self._logger.info(f"  '{name}' saved to: {outfile}")
            outfiles.append(outfile)

        # ── with-outliers companion CSVs ─────────────────────────────
        if self._show_outlier:
            with_outliers_dir = os.path.join(outdir, "with_outliers")
            if with_outliers_dir != "" and not os.path.exists(with_outliers_dir):
                os.makedirs(with_outliers_dir, exist_ok=True)
            for name in result_names:
                companion = f"{name}_with_outliers"
                if not hasattr(self, companion):
                    continue
                data = getattr(self, companion)
                if data is None:
                    continue
                outfile = os.path.join(
                    with_outliers_dir, companion.replace("_", "-") + ".csv"
                )
                data.to_csv(outfile, index=False)
                self._logger.info(f"  '{companion}' saved to: {outfile}")
                outfiles.append(outfile)

        # save plots
        for label, plot in self._pwc_plots.items():
            outfile = os.path.join(outdir, f"{label}.png")
            plot.savefig(outfile, bbox_inches='tight')
            self._logger.info(f"  Plot saved to: {outfile}")

    #-- data processing --#
    @wraps(ws_run_all)
    def run_all(self, *args, **kwargs) -> None:
        """
        Runs all data processing steps.
        """
        ws_run_all(self, *args, **kwargs)

    def refilter_events(
        self,
        filter_events: bool = True,
        min_event_amplitude: Optional[float] = 0.05,
        max_event_amplitude: Optional[float] = 10.0,
        min_event_fwhm: Optional[int] = 2,
        max_event_fwhm: Optional[int] = None,
        labels_corpus: Optional[Union[str, Path]] = None,
        on_disagreement: str = "drop",
        regenerate_plots: bool = False,
        regenerate_report: bool = False,
        output_dir: str = "wizards_staff_outputs",
        save_files: bool = False,
        show_plots: bool = False,
        p_th: float = 75.0,
        size_threshold: int = 20000,
        zscore_threshold: int = 3,
        percentage_threshold: float = 0.2,
    ) -> None:
        """
        Re-apply per-event filtering bounds to an already-processed Orb,
        without re-running spatial filtering, outlier detection, or per-event
        metric calculation.

        ``run_all`` stores the raw (pre-filter) per-event data on each shard;
        this method derives a fresh filtered view from those raw lists using
        the supplied bounds. Caches are invalidated for every event-filter-
        sensitive accessor so subsequent property reads return the new view:
        ``peak_amplitude_data``, ``max_peak_amplitude_data``, ``fwhm_data``,
        ``rise_time_data``, ``fall_time_data``, ``peak_to_peak_data``, and
        ``frpm_data``. ``well_peak_amplitude_data`` is derived live from
        ``peak_amplitude_data`` and updates automatically.

        ``outlier_data`` and PWC results are NOT touched — they are
        independent of per-event filter bounds.

        Args:
            filter_events: When False, all raw events are kept (bounds
                ignored). When True, the bounds below are applied.
            min_event_amplitude / max_event_amplitude: Inclusive bounds on
                peak ΔF/F. Set either to None to disable that side.
            min_event_fwhm / max_event_fwhm: Inclusive bounds on FWHM
                (frames). Set either to None to disable that side.
            labels_corpus: Optional path to a labels CSV produced by
                :class:`wizards_staff.labeling.event_labeler.EventLabeler`.
                When provided, events labeled ``"False"`` for matching
                ``(sample_id, roi_id, event_idx)`` are dropped from
                every per-event metric (third filter layer). This is the
                cheap path: run the analysis once with ``run_all``, label
                events with ``EventLabeler``, then call
                ``refilter_events(labels_corpus=...)`` to fold the labels
                into the analysis without redoing any heavy computation.
                Default ``None`` (no label-based filtering).
            on_disagreement: How to resolve events with conflicting
                labels from multiple labelers in the corpus. One of
                ``"drop"`` (default, precautionary), ``"keep"``, or
                ``"majority"`` (ties drop). Has no effect when
                ``labels_corpus`` is None.
            regenerate_plots: When True, re-emit the two filter-sensitive
                event-bar plots (``plot_sample_mean_dff_with_events`` and
                ``plot_neuron_dff_traces_with_events``). Default False so
                rapid parameter iteration stays cheap.
            regenerate_report: When True, re-run ``generate_run_report``
                with the new bounds. Default False.
            output_dir / save_files / show_plots: Forwarded to the plot /
                report calls when ``regenerate_plots`` or
                ``regenerate_report`` is True. Ignored otherwise.
            p_th, size_threshold, zscore_threshold, percentage_threshold:
                Forwarded to the event-bar plot calls. Should match the
                values used in the originating ``run_all`` call.
        """
        from wizards_staff.wizards.cauldron import _apply_event_filters
        from wizards_staff.plotting import (
            plot_sample_mean_dff_with_events,
            plot_neuron_dff_traces_with_events,
        )

        if filter_events:
            print(
                f'Per-event filtering ENABLED (filter_events=True): '
                f'amplitude in [{min_event_amplitude}, {max_event_amplitude}] '
                f'\u0394F/F, FWHM in [{min_event_fwhm}, {max_event_fwhm}] frames. '
                f'Re-deriving filtered views from raw events.',
                flush=True,
            )
        else:
            print(
                'Per-event filtering disabled (filter_events=False). '
                'All detected events are kept; min_event_* / max_event_* '
                'bounds are ignored.',
                flush=True,
            )

        if labels_corpus is not None:
            corpus_path = Path(labels_corpus)
            if corpus_path.exists():
                print(
                    f'Human-label filtering ENABLED: labels_corpus='
                    f'{str(corpus_path)!r}, on_disagreement={on_disagreement!r}. '
                    f'Events labeled False (and with no overriding True '
                    f'consensus) will be dropped from every per-event metric.',
                    flush=True,
                )
            else:
                print(
                    f'WARNING: labels_corpus {str(corpus_path)!r} does not '
                    f'exist; proceeding without label-based filtering.',
                    flush=True,
                )

        # Backward-compat bootstrap: pickled orbs from older versions don't
        # have the full set of _raw_*_data lists or the recording-level
        # scalars. Recover by copying the filtered lists into the raw
        # slots and pulling any missing peak positions from the filtered
        # peak-amplitude rows. Accurate when the prior run used
        # filter_events=False (no events were dropped); for prior
        # filter_events=True runs the bootstrap can only loosen, never
        # recover events that were dropped at run_all time. Surface that
        # caveat so users know.
        bootstrapped = []
        for shard in self._shards.values():
            for raw_attr in (
                '_raw_fwhm_data',
                '_raw_peak_amplitude_data',
                '_raw_rise_time_data',
                '_raw_fall_time_data',
                '_raw_peak_to_peak_data',
                '_raw_frpm_data',
            ):
                if not hasattr(shard, raw_attr):
                    setattr(shard, raw_attr, [])
            if not hasattr(shard, '_recording_n_frames'):
                shard._recording_n_frames = 0
            if not hasattr(shard, '_recording_frate'):
                shard._recording_frate = 0

            sample_name = shard.sample_name

            def _bootstrap(raw_attr, filtered_attr, extra_keys=()):
                raw = getattr(shard, raw_attr, None) or []
                filtered = getattr(shard, filtered_attr, None) or []
                if raw or not filtered:
                    return False
                copied = [dict(row) for row in filtered]
                setattr(shard, raw_attr, copied)
                return True

            did_bootstrap = False
            did_bootstrap |= _bootstrap('_raw_fwhm_data', '_fwhm_data')
            did_bootstrap |= _bootstrap(
                '_raw_peak_amplitude_data', '_peak_amplitude_data'
            )
            did_bootstrap |= _bootstrap('_raw_rise_time_data', '_rise_time_data')
            did_bootstrap |= _bootstrap('_raw_fall_time_data', '_fall_time_data')

            # peak_to_peak raw rows need 'Peak Positions' for
            # recomputation; pull them from the filtered peak amplitude
            # data when bootstrapping.
            if not shard._raw_peak_to_peak_data and getattr(
                shard, '_peak_to_peak_data', None
            ):
                pos_lookup = {
                    (row['Sample'], row['Neuron']): row.get('Peak Positions', [])
                    for row in (
                        getattr(shard, '_raw_peak_amplitude_data', None)
                        or getattr(shard, '_peak_amplitude_data', None)
                        or []
                    )
                }
                bootstrapped_p2p = []
                for row in shard._peak_to_peak_data:
                    bootstrapped_p2p.append({
                        **dict(row),
                        'Peak Positions': pos_lookup.get(
                            (row['Sample'], row['Neuron']), []
                        ),
                    })
                shard._raw_peak_to_peak_data = bootstrapped_p2p
                did_bootstrap = True

            # FRPM raw rows must carry the recording length so the new
            # event-rate definition can be recomputed. When the legacy
            # _frpm_data lacks N Frames / Frate, leave them at 0 — the
            # filtered FRPM will surface as NaN, which is more honest
            # than fabricating a rate.
            if not shard._raw_frpm_data and getattr(
                shard, '_frpm_data', None
            ):
                shard._raw_frpm_data = [
                    {
                        **dict(row),
                        'N Events': dict(row).get('N Events', 0),
                        'N Frames': dict(row).get(
                            'N Frames', shard._recording_n_frames
                        ),
                        'Frate': dict(row).get(
                            'Frate', shard._recording_frate
                        ),
                    }
                    for row in shard._frpm_data
                ]
                did_bootstrap = True

            if did_bootstrap:
                bootstrapped.append(sample_name)

        if bootstrapped:
            self._logger.warning(
                f"refilter_events: bootstrapped raw event lists from filtered "
                f"lists for {len(bootstrapped)} shard(s): "
                f"{', '.join(bootstrapped)}. This is accurate only if the "
                f"original run_all call used filter_events=False; if "
                f"filter_events=True was used, events dropped at run_all "
                f"time cannot be recovered (the bootstrap can only further "
                f"narrow the surviving events, not widen them)."
            )

        # Apply filters per shard.
        for shard in self._shards.values():
            try:
                _apply_event_filters(
                    shard,
                    filter_events=filter_events,
                    min_event_amplitude=min_event_amplitude,
                    max_event_amplitude=max_event_amplitude,
                    min_event_fwhm=min_event_fwhm,
                    max_event_fwhm=max_event_fwhm,
                    labels_corpus=labels_corpus,
                    on_disagreement=on_disagreement,
                )
            except Exception as e:
                self._logger.warning(
                    f"refilter_events failed for shard {shard.sample_name}: {e}"
                )

        # Invalidate orb-level cached DataFrames so the lazy _get_shard_data
        # path rebuilds them from the freshly-filtered shard lists. Every
        # event-filter-sensitive cache must be invalidated; outlier and PWC
        # caches are independent of event-filter bounds and stay valid.
        self._fwhm_data = None
        self._peak_amplitude_data = None
        self._max_peak_amplitude_data = None
        self._rise_time_data = None
        self._fall_time_data = None
        self._peak_to_peak_data = None
        self._frpm_data = None
        # Drop ledger is regenerated from scratch by every
        # ``_apply_event_filters`` call (per spec — the ledger always
        # reflects the *current* filter, never appended history), so
        # the orb-level cache must be invalidated alongside the other
        # event-filter-sensitive caches.
        self._event_drop_log = None

        if regenerate_plots:
            print(
                'Regenerating filter-sensitive event-bar plots...',
                flush=True,
            )
            for sh in list(self.shards):
                try:
                    plot_sample_mean_dff_with_events(
                        self,
                        sample_name=sh.sample_name,
                        exclude_outlier_neurons=True,
                        p_th=p_th,
                        size_threshold=size_threshold,
                        zscore_threshold=zscore_threshold,
                        percentage_threshold=percentage_threshold,
                        show_plots=show_plots,
                        save_files=save_files,
                        output_dir=output_dir,
                    )
                except Exception as e:
                    self._logger.warning(
                        f"plot_sample_mean_dff_with_events failed for "
                        f"{sh.sample_name}: {e}"
                    )
            try:
                plot_neuron_dff_traces_with_events(
                    self,
                    exclude_outlier_neurons=True,
                    p_th=p_th,
                    size_threshold=size_threshold,
                    show_plots=show_plots,
                    save_files=save_files,
                    output_dir=output_dir,
                )
            except Exception as e:
                self._logger.warning(
                    f"plot_neuron_dff_traces_with_events failed: {e}"
                )

        if regenerate_report:
            from wizards_staff.reporting import generate_run_report
            params = {
                "frate": None,
                "zscore_threshold": zscore_threshold,
                "percentage_threshold": percentage_threshold,
                "p_th": p_th,
                "size_threshold": size_threshold,
                "remove_outlier": self._remove_outlier,
                "show_outlier": self._show_outlier,
                "filter_events": filter_events,
                "min_event_amplitude": min_event_amplitude,
                "max_event_amplitude": max_event_amplitude,
                "min_event_fwhm": min_event_fwhm,
                "max_event_fwhm": max_event_fwhm,
                "labels_corpus": str(labels_corpus) if labels_corpus is not None else None,
                "on_disagreement": on_disagreement,
            }
            try:
                generate_run_report(
                    self,
                    params=params,
                    output_dir=output_dir,
                    save_files=save_files,
                )
            except Exception as e:
                self._logger.warning(
                    f"generate_run_report failed: {e}"
                )

    @wraps(ws_run_pwc)
    def run_pwc(self, *args, **kwargs) -> None:
        """
        Runs pairwise correlation analysis on all samples.
        """
        ws_run_pwc(self, *args, **kwargs)

    def save(self, outfile: str) -> None:
        """
        Saves the Orb object to disk via pickle.
        Args:
            outfile: Output file path.
        """
        # output directory
        outdir = os.path.dirname(outfile)
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # save object
        with open(outfile, 'wb') as f:
            pickle.dump(self, f)
        self._logger.info(f"Orb saved to: {outfile}")

    #-- misc --#
    def _repr_html_(self):
        return self.input_files.to_html()

    @staticmethod
    def _list_files(indir) -> List[str]:
        """
        Recursively lists all files in a directory.
        """
        files = []
        for dirpath, dirnames, filenames in os.walk(indir):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        return files

    #-- properties --#
    @property
    def samples(self):
        if self._samples is None:
            self._samples = set(self._shards.keys())
        return self._samples
    
    @property
    def num_shards(self):
        return len(self._shards)

    @property
    def shards(self):
        yield from self._shards.values()

    @property
    def input_files(self) -> pd.DataFrame:
        if self._input_files is None:
            try:
                self._input_files = pd.concat(
                    [shard.input_files for shard in self.shards]
                )
            except ValueError:
                logging.warning("No shards found")
        return self._input_files

    @property
    def input(self) -> pd.DataFrame:
        """
        Returns a DataFrame with all data items and file paths, merged with metadata.
        """
        if self._input is None:
            self._input = pd.merge(
                self.input_files.copy(), 
                self.metadata, on='Sample', how='left'
            )
        return self._input

    @property
    def results(self) -> Dict[str, pd.DataFrame]:
        """
        Return all results
        """
        yield from {
            'rise_time_data': self.rise_time_data,
            'fall_time_data': self.fall_time_data,
            'fwhm_data': self.fwhm_data,
            'frpm_data': self.frpm_data,
            'peak_amplitude_data': self.peak_amplitude_data,
            'max_peak_amplitude_data': self.max_peak_amplitude_data,
            'well_peak_amplitude_data': self.well_peak_amplitude_data,
            'peak_to_peak_data': self.peak_to_peak_data,
            'mask_metrics_data': self.mask_metrics_data,
            'silhouette_scores_data': self.silhouette_scores_data,
            'outlier_data': self.outlier_data,
            'df_mn_pwc': self.df_mn_pwc,
            'df_mn_pwc_intra': self.df_mn_pwc_intra,
            'df_mn_pwc_inter': self.df_mn_pwc_inter
        }.items()

    @property
    def rise_time_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with rise time data.

        When ``run_all`` was called with ``remove_outlier=True``, rows for
        outlier-flagged neurons are excluded. Use ``rise_time_data_with_outliers``
        for the unfiltered view.
        """
        DF = self._filter_outliers(self._get_shard_data('_rise_time_data'))
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['Rise Times', 'Rise Positions']        
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        # return after merging with metadata
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def rise_time_data_with_outliers(self) -> pd.DataFrame:
        """
        Returns the full ``rise_time_data`` DataFrame including outlier-flagged
        neurons (regardless of ``remove_outlier``).
        """
        DF = self._get_shard_data('_rise_time_data')
        if DF is None:
            return None
        cols = ['Rise Times', 'Rise Positions']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def fall_time_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with fall time data (time from peak to return to baseline).
        """
        DF = self._filter_outliers(self._get_shard_data('_fall_time_data'))
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['Fall Times', 'Fall Positions']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        # return after merging with metadata
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def fall_time_data_with_outliers(self) -> pd.DataFrame:
        """Full ``fall_time_data`` view including outlier-flagged neurons."""
        DF = self._get_shard_data('_fall_time_data')
        if DF is None:
            return None
        cols = ['Fall Times', 'Fall Positions']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def fwhm_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with FWHM data.
        """
        DF = self._filter_outliers(self._get_shard_data('_fwhm_data'))
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['FWHM Backward Positions', 'FWHM Forward Positions', 'FWHM Values', 'Spike Counts']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def fwhm_data_with_outliers(self) -> pd.DataFrame:
        """Full ``fwhm_data`` view including outlier-flagged neurons."""
        DF = self._get_shard_data('_fwhm_data')
        if DF is None:
            return None
        cols = ['FWHM Backward Positions', 'FWHM Forward Positions', 'FWHM Values', 'Spike Counts']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def frpm_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with FRPM data.
        """
        DF = self._filter_outliers(self._get_shard_data('_frpm_data'))
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def frpm_data_with_outliers(self) -> pd.DataFrame:
        """Full ``frpm_data`` view including outlier-flagged neurons."""
        DF = self._get_shard_data('_frpm_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def peak_amplitude_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with peak amplitude data (height of each calcium transient).
        """
        DF = self._filter_outliers(self._get_shard_data('_peak_amplitude_data'))
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['Peak Amplitudes', 'Peak Positions']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        # return after merging with metadata
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def peak_amplitude_data_with_outliers(self) -> pd.DataFrame:
        """Full ``peak_amplitude_data`` view including outlier-flagged neurons."""
        DF = self._get_shard_data('_peak_amplitude_data')
        if DF is None:
            return None
        cols = ['Peak Amplitudes', 'Peak Positions']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def max_peak_amplitude_data(self) -> pd.DataFrame:
        """
        Returns a per-neuron DataFrame with the largest peak amplitude observed
        on each neuron's trace (``Max Peak df/f``) and the standard deviation
        across that neuron's peak amplitudes (``Peak Amplitude Std``).

        Both values are derived from the same per-peak ΔF/F amplitudes that
        populate ``peak_amplitude_data`` (so units match: ΔF/F when raw ΔF/F
        was supplied to ``calc_peak_amplitude``, otherwise deconvolved units).

        Neurons with no detected peaks have ``Max Peak df/f = NaN`` and
        ``Peak Amplitude Std = NaN``. Neurons with exactly one detected peak
        have ``Peak Amplitude Std = 0``.

        When ``run_all`` was called with ``remove_outlier=True``, rows for
        outlier-flagged neurons are excluded.
        """
        DF = self._filter_outliers(self._get_shard_data('_max_peak_amplitude_data'))
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def max_peak_amplitude_data_with_outliers(self) -> pd.DataFrame:
        """Full ``max_peak_amplitude_data`` view including outlier-flagged neurons."""
        DF = self._get_shard_data('_max_peak_amplitude_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @staticmethod
    def _aggregate_peaks_by_well(peak_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Pool every detected peak across all neurons within each well and emit a
        per-well summary DataFrame.

        Input is the exploded ``peak_amplitude_data`` (or its _with_outliers
        variant) — one row per peak, with columns ``Sample``, ``Neuron``,
        ``Peak Amplitudes``, ``Well`` (from the metadata merge).

        Output columns:
            ``Well``
            ``Max Peak df/f``        (tallest peak observed in the well)
            ``Min Peak df/f``        (smallest peak observed in the well)
            ``Mean Peak df/f``       (mean across all peaks in the well)
            ``Median Peak df/f``     (median across all peaks in the well)
            ``Peak Amplitude Std``   (std across all peaks in the well)
            ``N Peaks``              (total number of detected peaks in the well)
            ``N Neurons``            (distinct (Sample, Neuron) pairs contributing)
        """
        if peak_df is None or peak_df.empty:
            return None
        if 'Well' not in peak_df.columns or 'Peak Amplitudes' not in peak_df.columns:
            return None

        # Peak Amplitudes came out of an explode(); cells may be object dtype, so coerce.
        df = peak_df.copy()
        df['Peak Amplitudes'] = pd.to_numeric(df['Peak Amplitudes'], errors='coerce')
        df = df.dropna(subset=['Peak Amplitudes'])
        if df.empty:
            return None

        agg = df.groupby('Well', dropna=False)['Peak Amplitudes'].agg(
            ['max', 'min', 'mean', 'median', 'std', 'count']
        ).rename(columns={
            'max':    'Max Peak df/f',
            'min':    'Min Peak df/f',
            'mean':   'Mean Peak df/f',
            'median': 'Median Peak df/f',
            'std':    'Peak Amplitude Std',
            'count':  'N Peaks',
        })

        # N Neurons: distinct (Sample, Neuron) pairs contributing peaks to the well.
        if 'Sample' in df.columns and 'Neuron' in df.columns:
            n_neurons = (
                df.drop_duplicates(['Sample', 'Neuron'])
                  .groupby('Well', dropna=False)
                  .size()
                  .rename('N Neurons')
            )
            agg = agg.join(n_neurons)
        else:
            agg['N Neurons'] = pd.NA

        return agg.reset_index()

    @property
    def well_peak_amplitude_data(self) -> pd.DataFrame:
        """
        Per-well summary of peak ΔF/F amplitudes, aggregated across **every
        detected peak from every neuron** in the well.

        Columns: ``Well``, ``Max Peak df/f``, ``Min Peak df/f``,
        ``Mean Peak df/f``, ``Median Peak df/f``, ``Peak Amplitude Std``,
        ``N Peaks``, ``N Neurons``.

        Respects ``remove_outlier``: when ``run_all`` was called with
        ``remove_outlier=True``, outlier-flagged neurons are excluded from
        the aggregation. Use ``well_peak_amplitude_data_with_outliers`` for
        the full view.
        """
        return self._aggregate_peaks_by_well(self.peak_amplitude_data)

    @property
    def well_peak_amplitude_data_with_outliers(self) -> pd.DataFrame:
        """
        Full per-well summary of peak ΔF/F amplitudes **including** outlier
        neurons (regardless of ``remove_outlier``).
        """
        return self._aggregate_peaks_by_well(self.peak_amplitude_data_with_outliers)

    @property
    def peak_to_peak_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with inter-spike interval (peak-to-peak distance) data.
        """
        DF = self._filter_outliers(self._get_shard_data('_peak_to_peak_data'))
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['Inter-Spike Intervals']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        # return after merging with metadata
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def peak_to_peak_data_with_outliers(self) -> pd.DataFrame:
        """Full ``peak_to_peak_data`` view including outlier-flagged neurons."""
        DF = self._get_shard_data('_peak_to_peak_data')
        if DF is None:
            return None
        cols = ['Inter-Spike Intervals']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def mask_metrics_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with mask metrics data.
        """
        DF = self._get_shard_data('_mask_metrics_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')
    
    @property
    def silhouette_scores_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with silhouette scores data.
        """
        DF = self._get_shard_data('_silhouette_scores_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def outlier_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with per-neuron outlier detection results
        (modified Z-score on max/mean/std of df/f).
        """
        DF = self._get_shard_data('_outlier_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def event_drop_log(self) -> pd.DataFrame:
        """Per-event drop ledger across all shards.

        Returns DataFrame with one row per dropped event. Columns:
        ``sample_id``, ``neuron_idx``, ``event_idx``, ``peak_amplitude``,
        ``fwhm_frames``, ``drop_reason``. ``peak_amplitude`` and
        ``fwhm_frames`` may be NaN/Inf — those are valid drop reasons
        recorded as ``drop_reason="nan_inf"``.

        The ledger is keyed by ``sample_id`` rather than the project's
        usual ``Sample`` column so it joins cleanly with the labels
        corpus written by
        :class:`wizards_staff.labeling.event_labeler.EventLabeler`.

        Returns an empty DataFrame (with the schema columns) when no
        events were dropped — useful for downstream code that wants to
        unconditionally read the columns. The result is rebuilt fresh
        whenever ``Orb.refilter_events`` invalidates the orb-level
        cache, so the ledger always reflects the current filter
        configuration.
        """
        cols = [
            "sample_id", "neuron_idx", "event_idx",
            "peak_amplitude", "fwhm_frames", "drop_reason",
        ]
        if self._event_drop_log is not None:
            return self._event_drop_log
        rows = []
        for shard in self.shatter():
            shard_log = getattr(shard, "_event_drop_log", None)
            if shard_log:
                rows.extend(shard_log)
        if rows:
            df = pd.DataFrame(rows, columns=cols)
        else:
            df = pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
        self._event_drop_log = df
        return df

    # pairwise correlations
    @property
    def df_mn_pwc(self) -> pd.DataFrame:
        """
        Returns a DataFrame with pairwise correlation results.
        """
        return self._df_mn_pwc

    @property
    def df_mn_pwc_intra(self) -> pd.DataFrame:
        """
        Returns a DataFrame with pairwise correlation results for intra-sample comparisons.
        """
        return self._df_mn_pwc_intra
    
    @property
    def df_mn_pwc_inter(self) -> pd.DataFrame:
        """
        Returns a DataFrame with pairwise correlation results for inter-sample comparisons.
        """
        return self._df_mn_pwc_inter

    @property
    def df_mn_pwc_all(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of pairwise correlation DataFrames.
        """
        return {
            'all': self.df_mn_pwc,
            'intra': self.df_mn_pwc_intra,
            'inter': self.df_mn_pwc_inter
        }

    #-- dunders --#
    def __str__(self) -> str:
        """
        Returns the input file summary table as a string
        """
        return self.input_files.to_string()

    __repr__ = __str__