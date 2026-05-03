import re
from .base import BaseValidator
class TractorValidator(BaseValidator):
    def isValid(self, text: str) -> bool:
        """Check if the document is valid for Tractor (Trac) report"""
        keywords = ["tractor", "trac", "fada", "oem", "sales", "market share"]
        text_lower = text.lower()
        return all(keyword in text_lower for keyword in keywords)

    def error_message(self) -> str:
        return "Document does not contain sufficient Tractor (Trac) segment data."