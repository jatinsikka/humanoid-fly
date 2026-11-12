"""Jet-powered humanoid flight tasks for Isaac Lab."""

from . import tasks  # noqa: F401  (registers the gym environments)


def register():
    """Hook for Isaac Lab's train/play scripts: ``--external_callback humanoid_fly.register``.

    Importing this package already registered the tasks. Returning None tells the script
    that this callback consumed no CLI arguments.
    """
    return None
