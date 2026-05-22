"""
Multi-image wrapper around :class:`~wizards_staff.labeling.event_labeler.EventLabeler`.

Wizards-Staff is built for biologists with limited Python experience: the
single-shard ``EventLabeler`` is a great primitive, but the natural
follow-up question — "now do that for every image in my dataset" — has
historically required a ``for`` loop, which doesn't compose with
ipywidgets (only the last widget renders, and the keyboard handlers
collide).

:class:`MultiShardLabeler` is the user-facing answer: one widget, one
cell, prev/next image buttons, an "image N of M" progress chip, an
auto-advance toggle, and a completion banner that names the very next
cell to run. The biologist never writes a loop and never thinks about
"shards" — the term is intentionally absent from the rendered UI in
favor of "image".

Internally the wrapper is a thin orchestrator around fresh
``EventLabeler`` instances, swapping the active child's root widget into
a single body container on each navigation. Persistence is unchanged
(every keystroke still hits the corpus CSV via ``EventLabeler._save``);
the wrapper just decides which shard's child labeler is in front.

Resume-on-reopen, auto-skip-empty-image, and per-image cursor
preservation are first-class behaviors here because biologists close
notebooks mid-session as a matter of course (lab meetings, day-over,
multi-day labeling pushes) and the labeler must come up *exactly where
they left off* without manual intervention.

This module is safe to import in headless environments. As with
:class:`EventLabeler`, ``ipywidgets`` is imported lazily inside
:meth:`MultiShardLabeler.display`.
"""

from __future__ import annotations

# import
## batteries
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
## package
from wizards_staff.labeling.event_labeler import EventLabeler

__all__ = ["MultiShardLabeler"]


class MultiShardLabeler:
    """
    Walk a labeler through every image (shard) in an Orb in one widget.

    The wrapper renders a small navigation header (image dropdown,
    prev / next buttons, auto-advance toggle, progress chip) above the
    currently-active :class:`EventLabeler` for the selected image. On
    next / prev / dropdown change, the wrapper:

    1. Snapshots the current child's view state (cursor, ROI cursor,
       view name) so revisits land where the labeler left off.
    2. Closes the child's matplotlib figure to free memory.
    3. Constructs a fresh :class:`EventLabeler` for the new shard and
       swaps its root widget into the body container.
    4. Restores the snapshot if one exists for that shard.

    Auto-advance: when ``auto_advance`` is True, the wrapper subscribes
    to the child's ``on_state_change`` hook and, after every
    label / skip / reject, advances to the next image whose unfinished
    count is positive. If every image is complete the wrapper switches
    to a completion banner that names the exact ``orb.refilter_events``
    cell to run next — biologists need an explicit "you're done" signal,
    not a frozen labeler on the last shard.

    Empty shards (no labelable events under the active Layer-2 bounds,
    e.g. every detection got filtered out by amplitude/FWHM caps) are
    silently skipped on next / prev navigation rather than rendering a
    confusing "no events on this shard" UI. The dropdown still lists
    them, marked with a leading bullet, so the labeler can manually
    inspect them if needed.

    Args:
        shards: Ordered iterable of Wizards-Staff Shard-like objects.
            Each must expose ``sample_name`` and the ``_raw_*`` attrs
            consumed by :class:`EventLabeler`. ``Orb.shatter()`` is the
            canonical producer.
        corpus_path: Path to the canonical labels CSV. Created on first
            label; shared across every shard's child labeler so all
            samples write into one corpus.
        labeler_id: Identifier for the human labeler. Forwarded to
            every child unchanged.
        context: Optional metadata dict (indicator, microscope,
            cell_type, experiment_id, sampling_rate). Forwarded to every
            child; appears on every corpus row.
        window_scale: Optional explicit per-event window width (in FWHM
            multiples). ``None`` (default) means the indicator-aware
            default. Forwarded to every child.
        ordering: Event ordering passed to each child. Defaults to the
            biologist-friendly ``"by_roi_then_time"``.
        auto_advance: When True (default), advance to the next
            unfinished image whenever the current one becomes complete.
            Surfaced as a dismissible toggle in the UI so the labeler
            can override per-session.
        start_at: Optional starting position. ``None`` (default) lands
            on the first unfinished image (resume behavior). An ``int``
            picks a specific dataset position; a ``str`` picks by
            ``sample_name``.
    """

    def __init__(
        self,
        shards: Iterable[Any],
        corpus_path: str,
        labeler_id: str,
        context: Optional[Dict[str, Any]] = None,
        window_scale: Optional[float] = None,
        ordering: str = "by_roi_then_time",
        auto_advance: bool = True,
        start_at: Optional[Union[int, str]] = None,
    ) -> None:
        shards_list = list(shards)
        if not shards_list:
            raise ValueError(
                "MultiShardLabeler requires at least one shard; got an empty "
                "iterable. Pass orb.shatter() (or any non-empty subset)."
            )

        # ``corpus_path`` is canonicalized here once so every child
        # labeler we construct downstream lands on the exact same path
        # regardless of cwd quirks between cells. Forwarding the raw
        # string would otherwise risk per-shard path drift if the
        # biologist passed a relative path and changed directories
        # mid-session.
        self._shards: List[Any] = shards_list
        self.corpus_path: str = os.path.abspath(corpus_path)
        self.labeler_id: str = labeler_id
        self.context: Dict[str, Any] = dict(context) if context else {}
        self.window_scale: Optional[float] = window_scale
        self.ordering: str = ordering
        self.auto_advance: bool = bool(auto_advance)

        self._logger = logging.getLogger(__name__)

        # Per-shard view-state snapshots (cursor, roi_cursor, view).
        # Keyed by sample_name so a labeler that bounces back to an
        # earlier image lands exactly where they left off — biologists
        # treat this as table-stakes, not a feature.
        self._cursor_state: Dict[str, Tuple[int, int, str]] = {}

        # Cheap (total_events, unfinished_count) cache keyed by
        # sample_name. Populated lazily on first probe and invalidated
        # for the active shard on every save. Without this, a single
        # navigation click iterates every shard to refresh the
        # progress chip, which (for a 50-shard dataset) walks 50 raw
        # event lists and reads the corpus 50 times. The cache cuts
        # that to one lookup per navigation. Concurrent writes by a
        # different labeler on shared storage are accepted as
        # eventually-consistent — we re-probe on the next session
        # restart, which matches the wrapper's "all your labels land
        # in one CSV regardless of who's labeling" model.
        self._probe_cache: Dict[str, Dict[str, int]] = {}

        # The currently active child labeler. Constructed lazily on
        # display() so import-time / construction-time of the wrapper
        # is cheap and exception-free for callers that just want to
        # introspect (e.g. test fixtures, programmatic dry runs).
        self._child: Optional[EventLabeler] = None

        # Cached widget handles populated on display(); used by tests
        # to drive button clicks without going through the keyboard.
        self._widgets: Dict[str, Any] = {}
        self._ipy = None  # ipywidgets module handle (lazy)

        # Resolved index of the active shard within self._shards.
        self._index: int = self._resolve_start_index(start_at)

        # Re-entrancy guard: while the wrapper is itself swapping
        # children (and therefore writing to the dropdown's value to
        # keep it in sync) we must not re-handle that change as if
        # the user clicked the dropdown.
        self._suppress_dropdown_observer: bool = False

        # Re-entrancy guard for auto-advance: the on_state_change
        # callback fires inside the child's _save(), which is itself
        # called during user actions; we want to schedule the advance
        # rather than swap children mid-call. Without this guard, an
        # advance triggered during reject_whole_trace's save would
        # tear down the very widgets the child is still writing to.
        self._pending_auto_advance: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _resolve_start_index(self, start_at: Optional[Union[int, str]]) -> int:
        """
        Resolve the constructor's ``start_at`` argument into an int index.

        ``None`` (default) means "first image with unfinished work" —
        the resume-on-reopen behavior. We compute this by constructing
        a transient child labeler for each shard until we find one
        whose ``unfinished_count`` is positive. This is more expensive
        than just landing on index 0, but it's only paid once per
        session and saves the labeler from hunting for "where was I?"
        — a real win for multi-day labeling pushes.
        """
        if start_at is None:
            for i, shard in enumerate(self._shards):
                stats = self._probe_shard(shard)
                if stats["unfinished"] > 0:
                    return i
            # Every shard is fully reviewed — start at 0 so the
            # completion banner has a deterministic anchor.
            return 0

        if isinstance(start_at, int):
            if not (0 <= start_at < len(self._shards)):
                raise IndexError(
                    f"start_at index {start_at} out of range "
                    f"[0, {len(self._shards)})"
                )
            return start_at

        if isinstance(start_at, str):
            for i, shard in enumerate(self._shards):
                if str(getattr(shard, "sample_name", "")) == start_at:
                    return i
            raise KeyError(
                f"start_at name {start_at!r} not found among shards "
                f"{[getattr(s, 'sample_name', None) for s in self._shards]!r}"
            )

        raise TypeError(
            f"start_at must be None, int, or str; got {type(start_at).__name__}"
        )

    def _make_child(self, shard: Any, *, quiet: bool = False) -> EventLabeler:
        """
        Construct a fresh :class:`EventLabeler` for ``shard``.

        ``quiet`` toggles the labeler's construction-time INFO logs
        (zero-event ROIs, "skipped N events outside Layer-2 bounds")
        between INFO (normal single-shard use, where this info is
        useful diagnostic output) and DEBUG (multi-shard probe path,
        where the same message firing once per shard is just noise).
        WARNING-level data-integrity logs are always emitted regardless.
        """
        return EventLabeler(
            shard=shard,
            corpus_path=self.corpus_path,
            labeler_id=self.labeler_id,
            context=self.context,
            window_scale=self.window_scale,
            ordering=self.ordering,
            quiet=quiet,
        )

    def _probe_shard(self, shard: Any) -> Dict[str, int]:
        """
        Cheap (total_events, unfinished_count) probe for ``shard``.

        Returns a cached dict if available; otherwise constructs a
        quiet :class:`EventLabeler`, populates the cache, and returns.
        Failures (a malformed shard, a corrupt corpus row) are swallowed
        with a single WARNING — we don't want one broken shard out of
        50 to take down the whole multi-image labeler at startup.
        """
        key = str(getattr(shard, "sample_name", id(shard)))
        cached = self._probe_cache.get(key)
        if cached is not None:
            return cached
        # If the active child is on this shard, prefer its in-memory
        # state — it can have unsaved view-state that's still tracked
        # in self._labels / self._trace_actions.
        if (
            self._child is not None
            and getattr(self._child, "shard", None) is shard
        ):
            result = {
                "total": self._child.total_events,
                "unfinished": self._child.unfinished_count,
            }
        else:
            try:
                probe = self._make_child(shard, quiet=True)
            except Exception as exc:
                self._logger.warning(
                    f"MultiShardLabeler: probe failed for shard "
                    f"{key!r} ({type(exc).__name__}: {exc}); reporting "
                    f"as zero events."
                )
                result = {"total": 0, "unfinished": 0}
            else:
                result = {
                    "total": probe.total_events,
                    "unfinished": probe.unfinished_count,
                }
        self._probe_cache[key] = result
        return result

    def _invalidate_probe(self, shard: Any) -> None:
        """Drop a cached probe entry, forcing the next read to re-walk."""
        key = str(getattr(shard, "sample_name", id(shard)))
        self._probe_cache.pop(key, None)

    # ------------------------------------------------------------------
    # Public read-only state
    # ------------------------------------------------------------------
    @property
    def n_shards(self) -> int:
        return len(self._shards)

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def current_shard(self) -> Any:
        return self._shards[self._index]

    @property
    def child(self) -> Optional[EventLabeler]:
        """The currently active :class:`EventLabeler`, or None pre-display()."""
        return self._child

    def progress_summary(self) -> Dict[str, Any]:
        """
        Cheap roll-up across every shard for the progress chip / tests.

        Uses :meth:`_probe_shard` so repeated calls on a stable
        dataset are O(1) per shard after the first. The active shard's
        cache entry is invalidated on every save (see the auto-advance
        wiring), so labels you've just made do show up in the progress
        chip on the next refresh — but a quiescent labeler doesn't
        re-walk every shard's raw events on every navigation click.
        """
        n_total_events = 0
        n_unfinished_events = 0
        n_reviewed_images = 0
        for shard in self._shards:
            stats = self._probe_shard(shard)
            n_total_events += stats["total"]
            n_unfinished_events += stats["unfinished"]
            if stats["unfinished"] == 0:
                # Both "fully labeled" (total > 0, unfinished == 0)
                # and "trivially complete" (total == 0) count as
                # reviewed. The latter is needed so the completion
                # banner can fire on datasets that contain empty
                # shards (e.g. wells where Layer-2 bounds dropped
                # every detection).
                n_reviewed_images += 1
        return {
            "n_images": len(self._shards),
            "n_reviewed_images": n_reviewed_images,
            "n_total_events": n_total_events,
            "n_labeled_events": n_total_events - n_unfinished_events,
            "all_complete": n_reviewed_images == len(self._shards),
        }

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def goto_image(self, target: Union[int, str]) -> None:
        """Jump to a specific image by index or sample_name."""
        if isinstance(target, str):
            for i, shard in enumerate(self._shards):
                if str(getattr(shard, "sample_name", "")) == target:
                    target = i
                    break
            else:
                raise KeyError(
                    f"sample_name {target!r} not found in this dataset"
                )
        if not isinstance(target, int):
            raise TypeError(
                f"goto_image expects int or str; got {type(target).__name__}"
            )
        if not (0 <= target < len(self._shards)):
            raise IndexError(
                f"image index {target} out of range [0, {len(self._shards)})"
            )
        if target == self._index:
            return
        self._switch_to(target)

    def next_image(self, *, skip_empty: bool = True) -> bool:
        """
        Advance to the next image. Returns True if it actually moved.

        When ``skip_empty`` is True (default), images with zero
        labelable events are silently skipped — biologists shouldn't
        see an "empty" UI when the underlying detection just had no
        events surviving the Layer-2 bounds.
        """
        new_idx = self._next_navigable_index(self._index, +1, skip_empty=skip_empty)
        if new_idx is None or new_idx == self._index:
            return False
        self._switch_to(new_idx)
        return True

    def prev_image(self, *, skip_empty: bool = True) -> bool:
        """Step back to the previous image. Returns True if it moved."""
        new_idx = self._next_navigable_index(self._index, -1, skip_empty=skip_empty)
        if new_idx is None or new_idx == self._index:
            return False
        self._switch_to(new_idx)
        return True

    def next_unfinished_image(self) -> bool:
        """
        Advance to the next image that still has unfinished events.

        Returns False if every later image (in dataset order) is
        already complete — that's the "everything is done" signal the
        wrapper uses to swap in the completion banner.
        """
        for i in range(self._index + 1, len(self._shards)):
            stats = self._probe_shard(self._shards[i])
            if stats["unfinished"] > 0:
                self._switch_to(i)
                return True
        return False

    def _next_navigable_index(
        self,
        start: int,
        step: int,
        *,
        skip_empty: bool,
    ) -> Optional[int]:
        """
        Walk ``start + step`` and beyond in ``step`` direction looking
        for the next index that the user should land on.

        ``skip_empty`` controls whether shards with zero labelable
        events are skipped over. ``None`` is returned when no valid
        index exists in the requested direction.
        """
        idx = start + step
        while 0 <= idx < len(self._shards):
            if not skip_empty:
                return idx
            # Empty-shard probe via the cache; the wrapper has already
            # walked every shard at startup, so subsequent navigation
            # never re-incurs that cost.
            stats = self._probe_shard(self._shards[idx])
            if stats["total"] > 0:
                return idx
            idx += step
        return None

    # ------------------------------------------------------------------
    # Switching machinery
    # ------------------------------------------------------------------
    def _snapshot_child(self) -> None:
        """Save the active child's view state for later restoration."""
        if self._child is None:
            return
        self._cursor_state[self._child.sample_id] = (
            int(getattr(self._child, "_cursor", 0)),
            int(getattr(self._child, "_roi_cursor", 0)),
            str(getattr(self._child, "_view", "overview")),
        )

    def _dispose_child(self) -> None:
        """Detach event handlers and free the child's matplotlib figure."""
        if self._child is None:
            return
        # Drop the auto-advance subscription before tearing anything
        # else down so a save in flight cannot trigger another switch.
        self._child.on_state_change = None
        fig = getattr(self._child, "_fig", None)
        if fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:  # pragma: no cover  — best-effort cleanup
                pass
        self._child = None

    def _restore_child_state(self, child: EventLabeler) -> None:
        """Apply any previously-snapshotted view state to ``child``."""
        snap = self._cursor_state.get(child.sample_id)
        if snap is None:
            return
        cursor, roi_cursor, view = snap
        if 0 <= cursor < len(child._events):
            child._cursor = cursor
        if 0 <= roi_cursor < len(child._rois_in_order):
            child._roi_cursor = roi_cursor
        if view in ("overview", "drill"):
            child._view = view

    def _switch_to(self, new_idx: int) -> None:
        """Tear down the current child and bring up ``new_idx``."""
        self._snapshot_child()
        self._dispose_child()
        self._index = new_idx

        # Build the new child, restore any prior view, and (if the
        # widget tree exists) swap it into the body container.
        new_shard = self._shards[new_idx]
        try:
            child = self._make_child(new_shard)
        except Exception as exc:
            self._logger.error(
                f"MultiShardLabeler: failed to construct labeler for "
                f"shard {getattr(new_shard, 'sample_name', '?')!r}: "
                f"{type(exc).__name__}: {exc}",
                exc_info=True,
            )
            raise
        self._restore_child_state(child)
        self._child = child
        self._wire_auto_advance(child)

        if self._widgets:
            self._render_active_view()
            self._refresh_header()

    def _wire_auto_advance(self, child: EventLabeler) -> None:
        """
        Subscribe to the child's state-change hook.

        Two responsibilities, neither optional:

        1. Invalidate the active shard's probe cache so the next
           ``progress_summary`` / header refresh reflects the label
           we just wrote. Without this, the dataset progress chip
           goes stale until the next image switch.
        2. Schedule an auto-advance to the next unfinished image when
           the active child becomes complete (only when
           ``self.auto_advance`` is True).

        The on_state_change hook is set unconditionally because the
        cache invalidation is correctness, not just UX polish.
        """
        def _on_state_change() -> None:
            self._invalidate_probe(child.shard)
            # Refresh the header in-place so the progress chip
            # updates immediately after a save without waiting for
            # the next nav click.
            try:
                self._refresh_header()
            except Exception as exc:  # pragma: no cover  — best-effort
                self._logger.debug(
                    f"MultiShardLabeler: header refresh after save "
                    f"failed ({type(exc).__name__}: {exc})"
                )

            if not self.auto_advance:
                return
            if not child.is_complete:
                return
            self._pending_auto_advance = True
            # We can't safely tear the child down mid-callback; the
            # matplotlib figure / widget refs are still live in the
            # outer call stack. Defer to the next event loop turn via
            # a lightweight scheduler — for the common Jupyter case
            # this just means scheduling on the IPython kernel's
            # asyncio loop. If no loop is running (e.g. headless tests)
            # we fall back to immediate execution; that path is safe
            # because the test harness drives the labeler
            # synchronously.
            self._schedule_auto_advance()

        child.on_state_change = _on_state_change

    def _schedule_auto_advance(self) -> None:
        """Run the pending auto-advance, asynchronously when possible."""
        try:
            import asyncio

            # ``get_event_loop()`` is deprecated when there is no
            # running loop; ``get_running_loop()`` raises in that
            # case but is the only safe API that won't create a new
            # loop in a thread that didn't expect one. The IPython
            # kernel always has a running loop in interactive use,
            # so this is the common production path.
            loop = asyncio.get_running_loop()
            loop.call_soon(self._run_pending_auto_advance)
            return
        except RuntimeError:
            # No running loop (typical for headless unit tests).
            pass
        except Exception:  # pragma: no cover  — fall through to sync
            pass
        # Execute synchronously. Safe when there's no live event
        # loop; the test harness drives the labeler synchronously
        # and any matplotlib figure / widget cleanup happens before
        # the call returns.
        self._run_pending_auto_advance()

    def _run_pending_auto_advance(self) -> None:
        if not self._pending_auto_advance:
            return
        self._pending_auto_advance = False
        # Re-check completion: a concurrent unrelated save (e.g. user
        # backed off the trace, or another callback path) could have
        # changed the picture between the schedule and the run.
        if self._child is None or not self._child.is_complete:
            return
        moved = self.next_unfinished_image()
        if not moved:
            # Every image is done — show the celebration / "what
            # next?" banner instead of leaving the labeler frozen on
            # the last shard with no obvious next action.
            self._render_completion_banner()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def display(self) -> None:
        """
        Render the multi-image labeling UI in the current notebook.

        Lazily imports ``ipywidgets`` and ``IPython.display`` so the
        module is safe to import in headless / batch environments.
        Returns ``None`` (matching :meth:`EventLabeler.display`) so the
        ipywidgets root isn't auto-rendered a second time as a cell
        return value.
        """
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "MultiShardLabeler.display() requires ipywidgets. Install "
                "with: pip install 'wizards_staff[labeling]'"
            ) from exc

        try:
            from IPython.display import display as _ipy_display
        except ImportError as exc:  # pragma: no cover  — IPython ships with notebooks
            raise ImportError(
                "MultiShardLabeler.display() requires IPython."
            ) from exc

        self._ipy = widgets

        # Header: vocabulary is "image", never "shard" — biologists
        # don't have a model for that internal term.
        title_html = widgets.HTML()
        progress_chip_html = widgets.HTML()

        prev_btn = widgets.Button(
            description="\u25c0 Prev image",
            tooltip="Go to the previous image with detectable events.",
        )
        next_btn = widgets.Button(
            description="Next image \u25b6",
            tooltip="Go to the next image with detectable events.",
        )
        next_unfinished_btn = widgets.Button(
            description="Next unfinished",
            tooltip=(
                "Jump to the next image that still has events you "
                "haven't reviewed."
            ),
        )

        dropdown = widgets.Dropdown(
            options=self._dropdown_options(),
            value=self._index,
            description="Image:",
            layout=widgets.Layout(width="55%"),
        )

        auto_advance_toggle = widgets.Checkbox(
            value=self.auto_advance,
            description="Auto-advance when image is complete",
            indent=False,
            tooltip=(
                "When on, the labeler jumps to the next unfinished "
                "image automatically as soon as you finish the "
                "current one. Turn off to dwell on one image."
            ),
        )

        body = widgets.VBox(
            [],
            layout=widgets.Layout(
                border="1px solid #ddd", padding="6px", margin="6px 0"
            ),
        )

        nav_row = widgets.HBox([prev_btn, next_btn, next_unfinished_btn, dropdown])
        toggle_row = widgets.HBox([auto_advance_toggle])

        # Permanent legend banner. The single-shard EventLabeler has
        # its own helpband banner inside the child; this wrapper-level
        # one explains the multi-image layer's mental model up front
        # so a biologist who's never seen the dataset before doesn't
        # have to infer it from button labels.
        wrapper_banner = widgets.HTML(
            value=(
                "<div style='font-size:0.85em;color:#333;background:#eef6ff;"
                "border:1px solid #b3d7ff;padding:6px 8px;border-radius:4px'>"
                "<b>Labeling across multiple images.</b> All your labels "
                "land in <b>one</b> shared corpus CSV regardless of which "
                "image you are on. You can stop and resume any time — "
                "the labeler reopens on the first image with unfinished "
                "events. When every image is reviewed, a banner here "
                "tells you the exact <code>orb.refilter_events(...)</code> "
                "call to run next."
                "</div>"
            )
        )

        root = widgets.VBox(
            [
                wrapper_banner,
                title_html,
                progress_chip_html,
                nav_row,
                toggle_row,
                body,
            ]
        )

        self._widgets = {
            "root": root,
            "title": title_html,
            "progress_chip": progress_chip_html,
            "prev_btn": prev_btn,
            "next_btn": next_btn,
            "next_unfinished_btn": next_unfinished_btn,
            "dropdown": dropdown,
            "auto_advance_toggle": auto_advance_toggle,
            "body": body,
            "wrapper_banner": wrapper_banner,
        }

        prev_btn.on_click(lambda _b: self.prev_image())
        next_btn.on_click(lambda _b: self.next_image())
        next_unfinished_btn.on_click(lambda _b: self.next_unfinished_image())

        def _on_dropdown_change(change: Dict[str, Any]) -> None:
            if self._suppress_dropdown_observer:
                return
            new_idx = change.get("new")
            if isinstance(new_idx, int) and new_idx != self._index:
                self.goto_image(new_idx)

        dropdown.observe(_on_dropdown_change, names="value")

        def _on_toggle_change(change: Dict[str, Any]) -> None:
            self.auto_advance = bool(change.get("new", False))
            if self._child is not None:
                self._wire_auto_advance(self._child)

        auto_advance_toggle.observe(_on_toggle_change, names="value")

        # Build the active child and stamp it into ``body``.
        if self._child is None:
            self._child = self._make_child(self._shards[self._index])
            self._restore_child_state(self._child)
            self._wire_auto_advance(self._child)
        self._render_active_view()
        self._refresh_header()

        _ipy_display(root)

    def _dropdown_options(self) -> List[Tuple[str, int]]:
        """Build the (label, index) pairs for the image dropdown."""
        opts: List[Tuple[str, int]] = []
        for i, shard in enumerate(self._shards):
            name = str(getattr(shard, "sample_name", f"image_{i}"))
            opts.append((f"{i + 1}/{len(self._shards)}  \u2014  {name}", i))
        return opts

    def _refresh_header(self) -> None:
        """Update the title + progress chip + dropdown selection."""
        if not self._widgets:
            return
        title = self._widgets["title"]
        chip = self._widgets["progress_chip"]
        dropdown = self._widgets["dropdown"]

        shard = self.current_shard
        sample_name = str(getattr(shard, "sample_name", f"image_{self._index}"))
        title.value = (
            f"<h3 style='margin:4px 0'>Image {self._index + 1} of "
            f"{self.n_shards} &mdash; <code>{sample_name}</code></h3>"
        )

        summary = self.progress_summary()
        chip.value = (
            f"<div style='font-size:0.9em;color:#333;background:#f6f6f6;"
            f"border:1px solid #ddd;padding:4px 8px;border-radius:4px;"
            f"display:inline-block'>"
            f"Reviewed: <b>{summary['n_reviewed_images']}/"
            f"{summary['n_images']}</b> images &middot; "
            f"<b>{summary['n_labeled_events']:,}</b>/"
            f"<b>{summary['n_total_events']:,}</b> events labeled by "
            f"<code>{self.labeler_id}</code>"
            f"</div>"
        )

        # Keep the dropdown in sync with the active index without
        # triggering its observer (which would cause re-entrancy).
        self._suppress_dropdown_observer = True
        try:
            dropdown.value = self._index
        finally:
            self._suppress_dropdown_observer = False

    def _render_active_view(self) -> None:
        """Render the active child's UI into the body container."""
        if not self._widgets:
            return
        body = self._widgets["body"]
        if self._child is None:
            body.children = ()
            return
        # Build the child's widget tree WITHOUT calling its display()
        # method. ``_build_root_widget`` constructs the same tree that
        # ``EventLabeler.display`` would, but stops short of the
        # ``IPython.display.display(root)`` call. The wrapper owns the
        # only render path (we set body.children below); calling the
        # child's display() here would render the child's tree
        # standalone in the notebook output AND embed it inside our
        # body container, producing the duplicated-UI bug.
        if "root" not in self._child._widgets:
            self._child._build_root_widget()
        body.children = (self._child._widgets["root"],)

    def _render_completion_banner(self) -> None:
        """
        Replace the active child's view with a "you're done" banner
        that names the exact ``orb.refilter_events`` call to run next.

        Biologists need an explicit transition; otherwise the labeler
        sits frozen on the last image and they're left wondering "did
        it save? did it work? what now?". The banner converts that
        ambiguity into a single concrete action.
        """
        if not self._widgets:
            return
        widgets = self._ipy
        if widgets is None:
            return
        body = self._widgets["body"]

        # Free the child labeler — it's complete and we don't need
        # to keep its matplotlib figure resident.
        self._dispose_child()

        review_btn = widgets.Button(
            description="Review again from the top",
            tooltip=(
                "Re-open the first image. Useful if you want to "
                "double-check your work before re-running "
                "refilter_events."
            ),
        )

        def _on_review(_b: Any) -> None:
            self.goto_image(0)

        review_btn.on_click(_on_review)

        banner_html = widgets.HTML(
            value=(
                "<div style='font-size:1em;color:#0a662e;background:#e6f6ec;"
                "border:1px solid #7ec18f;padding:14px 16px;border-radius:6px'>"
                f"<b>\u2713 All {self.n_shards} image"
                f"{'s' if self.n_shards != 1 else ''} reviewed by "
                f"<code>{self.labeler_id}</code> \u2014 labeling complete!</b>"
                "<br/><br/>"
                "Next, fold these labels into your downstream analysis with:"
                f"<pre style='background:#f6f6f6;padding:8px;border-radius:4px;"
                f"font-size:0.95em;margin-top:6px'>"
                f"orb.refilter_events(\n"
                f"    labels_corpus={self.corpus_path!r},\n"
                f"    on_disagreement=\"drop\",\n"
                f")</pre>"
                "<span style='color:#555'>"
                "Re-running this notebook's <code>refilter_events</code> "
                "cell now will drop every event you marked <code>False</code> "
                "(plus any whole-trace rejections) from every per-event "
                "metric."
                "</span>"
                "</div>"
            )
        )

        body.children = (banner_html, review_btn)
        self._refresh_header()
