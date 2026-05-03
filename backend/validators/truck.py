from .base import BaseValidator
class TruckValidator(BaseValidator):
    def isValid(self, text: str) -> bool:
        """Check if the document is valid for Truck report"""
        keywords = ["truck", "commercial vehicle", "fada", "oem", "sales", "market share"]
        text_lower = text.lower()
        return all(keyword in text_lower for keyword in keywords)

    def error_message(self) -> str:
        return "Document does not contain sufficient Truck segment data."