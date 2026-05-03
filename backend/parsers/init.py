from .tractor import TractorParser
from .truck import TruckParser  
from .brandwise import BrandWiseParser
def get_parser(report_type: str):
    parsers = {
        "tractor": TractorParser(),
        "truck": TruckParser(),
        "brandwise": BrandWiseParser(),
    }.get(report_type)
    return parsers