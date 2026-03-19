"""
Alternate entry point — delegates to the top-level main.py.
"""
import sys
from pathlib import Path

# Add parent directory so the top-level main.py's imports work
_WHISPER_APP_ROOT = str(Path(__file__).resolve().parent.parent)
if _WHISPER_APP_ROOT not in sys.path:
    sys.path.insert(0, _WHISPER_APP_ROOT)

from main import main  # noqa: E402

if __name__ == "__main__":
    main()