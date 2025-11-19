# streamlit_app.py

import streamlit as st
import pdfplumber
import json
import re
import os
import base64
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, Response
from playwright.sync_api import sync_playwright
import markdown

# -------------------------------
# Groq client setup
# -------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACT_MODEL = "llama-3.1-8b-instant"  # fact extraction model
GEN_MODEL = "llama-3.3-70b-versatile"                    # article generation model

# -------------------------------
# Helper Functions
# -------------------------------

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


def extract_text_from_pdf(uploaded_file):
    """Extracts plain text from PDF file"""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def extract_facts(report_text: str):
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


def generate_article(facts, sample_texts, brand_map):
    """Generate article in writer's style using Groq"""
    sample_block = "\n\n".join(sample_texts[:2])  # take 2 samples for context

    prompt = (
    "You are a professional farming and tractor industry journalist who writes in a clear, data-driven, and factual tone "
    "similar to the provided sample articles (e.g., FADA August 2025 report). Your task is to generate a detailed tractor "
    "industry article using ONLY the factual data provided.\n\n"

    "=== WRITING STYLE ===\n"
    "- Follow the structure and tone of the sample report: concise, factual, and professionally journalistic.\n"
    "- Use short paragraphs and a positive yet balanced tone.\n"
    "- Every OEM (including 'Others' and 'Total') must appear in both the table AND the analysis paragraphs.\n"
    "- Sentences should smoothly transition between facts, highlighting YoY changes, leadership, and trends.\n"
    "- Avoid speculation, fluff, or marketing language.\n\n"

    "=== STRUCTURE ===\n"
    "1. Begin with a **headline** that is positive and relevant in H1 font (e.g., 'FADA Retail Sales Report August 2025: Industry records 85,215 Units with 30.14% Growth').\n"
    "2. Add an **introductory paragraphs** summarizing overall tractor (Trac) industry performance\n"
    "(e.g.- 6 June 2025: The Federation of Automobile Dealers Associations (FADA) has released retail sales data for the tractor segment for May 2025."
     "As per the latest figures, the Indian tractor industry recorded total sales of 71,992 units, marking a 2.75% growth over 70,063 units sold in May 2024.\n"
     "Despite varied performances across leading OEMs, the industry managed to stay in positive territory." 
     "This reflects steady rural demand and the early impact of seasonal preparation.)\n"
     "NOTE: ALL THE CRUCIAL DATA LIKE <DATE>, <SALES FIGURES>, ETC., MUST BE TAKEN FROM THE FACTUAL DATA PROVIDED BELOW AND SHOULD BE IN BOLD.\n"
     "THE PUBLISHING DATE SHOULD BE EXACT and MENTION SALES DRIVING FACTOR ONLY IF MENTIONED IN FACTS\n\n"
    "3. Insert a **Tractor OEM Performance Table** (in markdown format) with the following columns:\n"
    "   - OEM Name\n"
    "   - <MONTH> <CURRENT YEAR> Sales (e.g., August 2025)\n"
    "   - <MONTH> <PREVIOUS YEAR> Sales (e.g., August 2024)\n"
    "   - YoY Sales Growth (%)\n"
    "   - <MONTH> <CURRENT YEAR> Market Share (%)\n"
    "   - <MONTH> <PREVIOUS YEAR> Market Share (%)\n"
    "   - YoY Market Share Growth (%)\n\n"
    "   The table must include all brands present in the data, including 'Others' and 'Total'.\n\n"

    "4. Write **Brand-Wise Tractor Sales Performance paragraphs explaining each OEM individually – <MONTH> <CURRENT YEAR>**, covering each OEM (including others) in order of sales:\n"
    "<OEM NAME>, <CURRENT YEAR SALES UNIT>, <CURRENT YEAR DATE>, <LAST YEAR SALES UNIT>, <LAST YEAR DATE>, <YOY UNIT GROWTH>,  <YOY UNIT GROWTH>in percentage,"
    "<CURRENT YEAR MARKET SHARE>, <LAST YEAR MARKET SHARE> <YOY MARKET SHARE CHANGE>. Highlight leaders, significant gainers, and any noteworthy trends."
    "(e.g., -Mahindra & Mahindra Ltd (Tractor Division) led the segment with 16,511 units sold in May 2025, up from 15,921 units in the same month last year."
    " The company registered a 3.71% sales growth." 
    "Its market share rose from 22.72% in May 2024 to 22.93% in May 2025, marking a 0.21% gain.)\n"
    "NOTE: ALL THE CRUCIAL DATA LIKE NAMES < SALES FIGURES >, ETC., MUST BE TAKEN FROM THE FACTUAL DATA PROVIDED BELOW AND SHOULD BE IN BOLD.\n"
    "EACH OEM SHOULD HAVE A SEPERATE PARAGRAPH FOR ITS SALES PERFORMANCE\n\n"

    "5. Write a POSITIVE or NEUTRAL**summary paragraph** analyzing total performance (the 'Total' row) and comparing overall market sentiment "
    "with the previous year. Also mention the reason of in the change in sales if mentioned."
    "(e.g., festive demand, good monsoon, subsidies)\n"


    "=== FACTUAL DATA (Tractor Segment Only) ===\n"
    f"{json.dumps(facts, indent=2)}\n\n"

    "=== BRAND MAP ===\n"
    f"{json.dumps(brand_map, indent=2)}\n\n"

    "=== RULES ===\n"
    "- Use ONLY the above factual data — no external or inferred information.\n"
    "- Mention every OEM listed, including 'Others' and 'Total'.\n"
    "- Keep the tone positive, informative, and aligned with the sample document style.\n"
    "- Ensure numerical accuracy (percentages and units must match the data exactly).\n"
    )

    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional farming and tractor industry journalist"},
            {"role": "user", "content": prompt}
        ],
        model=GEN_MODEL,
        max_completion_tokens=2500,
        temperature=0
    )

    return resp.choices[0].message.content


def replace_brand_names(text, mapping):
    """Replace company names with brand names"""
    for old, new in mapping.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text



    """Convert markdown article to PDF bytes using Playwright."""
    html = markdown.markdown(article_md)

    # Add optional styling
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: "Helvetica", sans-serif;
                line-height: 1.5;
                font-size: 12pt;
                padding: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 16px 0;
            }}
            table, th, td {{
                border: 1px solid #333;
                padding: 8px;
            }}
            th {{
                background: #f0f0f0;
            }}
            h1, h2, h3 {{
                color: #0A66C2;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")

        pdf_bytes = page.pdf(
            format="A4",
            margin={"top":"12mm","bottom":"12mm","left":"12mm","right":"12mm"}
        )

        browser.close()
        return pdf_bytes



# -------------------------------
# Streamlit UI
# -------------------------------
st.image("cover.png", width='stretch')
st.set_page_config(page_title="AI Writer", layout="wide")
st.title("The AI Writer")

# Upload section
report_file = st.file_uploader("Upload Sales Report (PDF)", type=["pdf"])
sample_files = st.file_uploader("Upload Sample Articles (PDF)", type=["pdf"], accept_multiple_files=True)
# Brand map
brand_map = {}
df = pd.read_csv("Data/BrandMap.csv")
brand_map = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

# Main button
if st.button("Generate Article"):
    if not report_file or not sample_files:
        st.error("Please upload sales report and sample articles.")
    else:
        with st.spinner("Reading PDFs..."):
            report_text = extract_text_from_pdf(report_file)
            sample_texts = [extract_text_from_pdf(f) for f in sample_files]

        

        with st.spinner("Extracting facts..."):
            facts=get_table_json_by_header(report_file,"Tractor OEM")
            #facts = extract_facts(report_text)
        #st.json(facts)

        with st.spinner("Generating article..."):
            article = generate_article(facts, sample_texts, brand_map)
            article = replace_brand_names(article, brand_map)

        st.markdown("### 📰 Generated Article")
        st.write(article)