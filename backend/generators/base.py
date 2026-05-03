class BaseWriter:
    def generate(self, dataset: dict) -> str:
        raise NotImplementedError("Subclasses must implement this method")