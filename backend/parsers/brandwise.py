class TruckParser:
    def parse(self, text: str) -> dict:
        return {
            "report_type": "brand_segment",
            "segment": "Brand",
            "period": None,
            "oem_performance": [],
            "total_sales": None
        }