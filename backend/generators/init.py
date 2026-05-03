from .tractor import TractorWriter
from .truck import TruckWriter
from .brandwise import BrandWiseWriter

def get_writer(report_type: str):
    return {
        "tractor_segment": TractorWriter(),
        "truck_segment": TruckWriter(),
        "brand_wise": BrandWiseWriter()
    }.get(report_type)
