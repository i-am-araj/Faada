class TruckParser:
    def parse(self, text: str) -> dict:
        return {
            "report_type": "truck_segment",
            "segment": "Truck",
            "period": None,
            "oem_performance": [],
            "total_sales": None
        }