from .tractor import TractorValidator
from .truck import TruckValidator
from .brandwise import BrandWiseValidator

def get_validators(report_type: str):
    validators = {
        "tractor": TractorValidator(),
        "truck": TruckValidator(),
        "brandwise": BrandWiseValidator(),
    }.get(report_type)