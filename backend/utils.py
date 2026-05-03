import pdfplumber
import pandas as pd

def extract_text_from_pdf(uploaded_file):
    """Extracts plain text from PDF file"""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()

def get_table_json_by_header(pdf_file: str, keyword: str):
    """
    Retrieve the first table in a PDF whose header contains a given keyword.
    Returns table data as a JSON string.
    """
    keyword = keyword.lower()

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables() or []

            for tbl in extracted:
                # tbl = list of lists
                if not tbl or len(tbl) < 2:
                    continue

                header = tbl[0]
                rows   = tbl[1:]

                # Normalize to lowercase for searching
                header_lower = [h.lower() if h else "" for h in header]

                if any(keyword in h for h in header_lower):
                    # Build DataFrame
                    df = pd.DataFrame(rows, columns=header)

                    # Convert to JSON string (records format)
                    json_str = df.to_json(orient="records", force_ascii=False)

                    return json_str   # ✅ return JSON string

    raise ValueError(f"No table found whose header contains keyword '{keyword}'.")

def extract_tractor_facts(report_text: str):
    """Use Groq to extract structured Tractor (Trac) JSON facts safely."""
    
    short_instructions = (
        "Extract only Tractor (Trac) segment dataa and publishing date from the report below. "
        "Ignore other segments. Return strictly valid JSON (no commentary). "
        "Schema:\n"
        "{ 'segment': 'Tractor', 'period': '<Month Year>', 'total_sales': <int>, "
        "'previous_year_sales': <int>, 'yoy_growth_percent': <float>, "
        "'oem_performance': [ {'brand': <str>, 'sales': <int>, 'previous_sales': <int>, "
        "'growth_units': <int>, 'growth_percent': <float>, 'market_share_percent': <float>} ], "
        "'notes': <short summary> }"
    )

    # Trim very long input
    report_snippet = report_text[-8000:] if len(report_text) > 8000 else report_text

    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": short_instructions},
            {"role": "user", "content": report_snippet}
        ],
        model="llama-3.1-8b-instant",
        max_completion_tokens=512,
        temperature=0.0
    )

    content = resp.choices[0].message.content.strip()
    m = re.search(r"(\{[\s\S]*\})", content)
    if not m:
        return {}

    json_text = m.group(1)

    # 🧠 JSON Repair Step
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Try to auto-fix common issues like trailing commas or missing brackets
        fixed = (
            json_text
            .replace("\n", "")
            .replace("\r", "")
            .replace(",}", "}")
            .replace(",]", "]")
        )
        # Remove duplicate commas
        fixed = re.sub(r",\s*,", ",", fixed)
        try:
            return json.loads(fixed)
        except Exception:
            print("⚠️ JSON repair failed. Raw output returned for debugging.")
            return {"raw_output": json_text}
