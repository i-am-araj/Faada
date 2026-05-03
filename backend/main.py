from fastapi import FastAPI, UploadFile, File, HTTPException
from utils import extract_text_from_pdf
from validators import get_validators
from parsers import get_parser
from generators import get_writer

app = FastAPI(title="Sales Report Article Generator")

@app.get("/generate")
def generate_report(report_type: str, report_file: UploadFile = File(...)):
    text=extract_text_from_pdf(report_file.file)
    validator = get_validators(report_type)

    if not validator:
        raise HTTPException(status_code=400, detail="Invalid report type")
    
    if not validator.is_valid(text):
        raise HTTPException(400, validator.error_message())
    
    parser= get_parser(report_type)
    dataset=parser.parse(text)

    writer = get_writer(report_type)
    article=writer.generate(dataset)

    return {
        "report_type": report_type,
        "article": article
    }