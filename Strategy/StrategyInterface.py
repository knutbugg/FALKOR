
class Strategy:
    def prepare(self):
        """Perform any initial computations. Train a nn, etc. """
        raise NotImplementedError

    def generate_signals(self):
        """Generate all signals from the current data"""
        raise NotImplementedError

    def update(self, update_info):
        """Update the corresponding data files with any changes """
        raise NotImplementedError
