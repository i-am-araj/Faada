# streamlit_app.py

import streamlit as st
import pdfplumber
import json
import re
import os
import base64
import subprocess
import sys
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
        "Extract only Tractor (Trac) segment data from the report below. "
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
    "1. Begin with a **headline** that is positive and relevant (e.g., 'FADA Retail Tractor Sales Surge 30% in August 2025').\n"
    "2. Add an **introductory paragraph** summarizing overall tractor (Trac) performance — total sales, YoY growth, "
    "and drivers such as rural demand or monsoon influence if available.\n"
    "3. Insert a **Tractor OEM Performance Table** (in markdown format) with the following columns:\n"
    "   - OEM Name\n"
    "   - Current Period Sales (e.g., August 2025)\n"
    "   - Previous Year Sales (e.g., August 2024)\n"
    "   - YoY Sales Growth (%)\n"
    "   - Current Year Market Share (%)\n"
    "   - Previous Year Market Share (%)\n"
    "   - YoY Market Share Growth (%)\n\n"
    "   The table must include all brands present in the data, including 'Others' and 'Total'.\n\n"
    "4. Write **brand-wise performance analysis paragraphs**, covering each OEM in order of sales:\n"
    "   - Mention their sales volume, YoY change (in % and units), and market share.\n"
    "   - For each, briefly describe whether their market share rose, fell, or remained stable.\n"
    "   - End with a sentence summarizing that OEM’s trend or positioning.\n\n"
    "5. Write a **summary paragraph** analyzing total performance (the 'Total' row) and comparing overall market sentiment "
    "with the previous year.\n"
    "6. Conclude with a **positive outlook** paragraph summarizing future prospects (e.g., festive demand, good monsoon, subsidies).\n\n"

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
        max_completion_tokens=1500,
        temperature=0.7
    )

    return resp.choices[0].message.content


def replace_brand_names(text, mapping):
    """Replace company names with brand names"""
    for old, new in mapping.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def article_to_pdf(article_text, filename="article.pdf"):
    """Save article text to PDF with ReportLab"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in article_text.splitlines():
        if not line.strip():
            y -= 20
            continue
        c.drawString(50, y, line.strip())
        y -= 15
        if y < 50:  # new page
            c.showPage()
            y = height - 50
    c.save()
    return filename

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="AI Writer Tool", layout="wide")
st.title("✍️ AI Writer Tool")

# Upload section
report_file = st.file_uploader("📊 Upload Sales Report (PDF)", type=["pdf"])
sample_files = st.file_uploader("📝 Upload Sample Articles (PDF)", type=["pdf"], accept_multiple_files=True)
brand_map_file = st.file_uploader("🏷 Upload Company Mapping (CSV)", type=["csv"])

# Main button
if st.button("🚀 Generate Article"):
    if not report_file or not sample_files:
        st.error("Please upload sales report and sample articles.")
    else:
        with st.spinner("Reading PDFs..."):
            report_text = extract_text_from_pdf(report_file)
            sample_texts = [extract_text_from_pdf(f) for f in sample_files]

        # Brand map
        brand_map = {}
        if brand_map_file:
            df = pd.read_csv(brand_map_file)
            brand_map = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        with st.spinner("Extracting facts..."):
            facts=get_table_json_by_header(report_file,"Tractor OEM")
            #facts = extract_facts(report_text)
        #st.json(facts)

        with st.spinner("Generating article..."):
            article = generate_article(facts, sample_texts, brand_map)
            article = replace_brand_names(article, brand_map)

        st.markdown("### 📰 Generated Article")
        st.write(article)

        with st.spinner("Creating PDF..."):
            pdf_fn = article_to_pdf(article, "final_article.pdf")
            with open(pdf_fn, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="article.pdf">📥 Download PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
