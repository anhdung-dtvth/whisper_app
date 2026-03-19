# app/__init__.py
"""
WhisperSign Application Package.

WHY THIS FILE?
- Makes 'app' directory a Python package
- Allows: from app.hardware import MockController
- Without this: ImportError!

METAPHOR:
- app/ is a building
- __init__.py is the front door
- Other files are rooms inside
"""

__version__ = "0.1.0"
__author__ = "Anh Dung"
__description__ = "Real-time Sign Language Recognition"

# Print when package is imported (debugging)
print(f"[WhisperSign] Loading v{__version__}")


# ============================================================================
# Package-level imports (optional, for convenience)
# ============================================================================

# Users can do: from app import MockController
# Instead of: from app.hardware.mock_controller import MockController

# But be careful: imports here affect ALL files!
# Recommendation: Keep minimal for now

# Example (commented out for now):
# from .hardware.mock_controller import MockController
# from .services.model_service import ModelService