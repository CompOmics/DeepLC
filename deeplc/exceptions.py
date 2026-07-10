"""DeepLC exceptions."""


class DeepLCError(Exception):
    """Base class for DeepLC exceptions."""


class CalibrationError(DeepLCError):
    """Raised when calibration fails."""


class ReferenceSelectionError(DeepLCError):
    """Raised when reference PSM selection fails."""
