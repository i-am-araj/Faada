from .base import BaseValidator

class BrandWiseValidator(BaseValidator):
    def is_valid(self, text: str) -> bool:
        keywords = ["mahindra", "tata", "ashok", "eicher"]
        return any(k in text.lower() for k in keywords)

    def error_message(self) -> str:
        return "This document does not contain brand-wise sales data."