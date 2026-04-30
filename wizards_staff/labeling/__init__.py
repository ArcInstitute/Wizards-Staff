"""
Hand-labeling tools for Wizards-Staff calcium event detections.

This subpackage is safe to import in headless environments. The interactive
:class:`EventLabeler` itself only requires ``ipywidgets`` once
:meth:`EventLabeler.display` is called; that import is performed lazily so
that callers (e.g. the Lizard-Wizard CLI invoking Wizards-Staff in batch
mode) do not need ``ipywidgets`` installed merely to import the package.

Install the optional widget dependency with::

    pip install 'wizards_staff[labeling]'
"""

from wizards_staff.labeling.event_labeler import EventLabeler

__all__ = ["EventLabeler"]
