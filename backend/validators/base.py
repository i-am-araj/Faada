class BaseValidator:
    def isValid(self, text: str) -> bool:
        raise NotImplementedError("Subclasses should implement this method")
    def error_message(self) -> str:
        return "Document not valid for this report type."