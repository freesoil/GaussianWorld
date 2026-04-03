"""Resolve dataset roots from environment (optional overrides for local layouts)."""
import os


def dummy_surroundocc_enabled() -> bool:
    return os.environ.get('GAUSSIANWORLD_DUMMY_SURROUNDOCC', '').lower() in (
        '1', 'true', 'yes',
    )


def resolve_under_data_root(data_path: str, filepath: str) -> str:
    """Join *data_path* if *filepath* is not absolute (nuScenes assets are usually relative)."""
    if not filepath:
        return filepath
    if os.path.isabs(filepath):
        return filepath
    return os.path.normpath(os.path.join(data_path, filepath))


def resolve_nuscenes_data_path(configured_path: str) -> str:
    """If NUSCENES_DATA_ROOT is set, use it as the nuScenes root instead of *configured_path*."""
    root = os.environ.get('NUSCENES_DATA_ROOT')
    if not root:
        return configured_path
    root = os.path.expanduser(root)
    if not root.endswith(os.sep):
        root = root + os.sep
    return root
