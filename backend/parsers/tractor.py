class TractorParser:
    def parse(self, text: str) -> dict:
        return {
            "report_type": "tractor_segment",
            "segment": "Tractor",
            "period": None,
            "oem_performance": [],
            "total_sales": None
        }