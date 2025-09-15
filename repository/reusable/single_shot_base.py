from ndscan.experiment import ExpFragment, ResultChannel

class SingleShotBase(ExpFragment):
    """One physical attempt. Subclasses must expose their output channels."""

    def get_classification_handle(self) -> ResultChannel:
        """Return the 0/1 (bright) classification channel handle."""
        raise NotImplementedError

    def get_counts_handle(self) -> ResultChannel:
        """Return the integer counts channel handle."""
        raise NotImplementedError