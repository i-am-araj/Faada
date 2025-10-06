import re
import numpy as np
import pandas as pd
import pdfplumber
from collections import defaultdict

# ----------------------------
# 📌 Extract Tables
# ----------------------------
def extract_tables_from_pdf(pdf_file):
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])  # first row as header
                    tables.append(df)
    return tables

# ----------------------------
# 📌 Year + Column Utilities
# ----------------------------
_month_names = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)"

def extract_year_from_text(s: str):
    if not isinstance(s, str): return None
    s = s.strip()
    m = re.search(r"(20\d{2})", s)   # 2025
    if m: return int(m.group(1))
    m = re.search(r"FY\s*['-]?\s*(\d{2,4})", s, re.I)  # FY25
    if m:
        y = int(m.group(1))
        return y if y > 1000 else 2000 + y
    m = re.search(rf"{_month_names}.*?(20\d{{2}})", s, re.I)  # July 2025
    if m: return int(m.group(2))
    m = re.search(r"[’'`]\s*(\d{2})", s)  # '25
    if m: return 2000 + int(m.group(1))
    return None

def clean_numeric_cell(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s in {"-", "—", "na", "n/a", "nil"}: return np.nan
    neg = False
    if re.match(r"^\(.*\)$", s):
        neg, s = True, s.strip("()")
    s = s.replace(",", "").replace("%", "")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    if not m: return np.nan
    val = float(m.group(0))
    return -val if neg else val

def extract_metric_name(colname: str):
    s = str(colname)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"(20\d{2})", " ", s)
    s = re.sub(r"[’'`]\s*\d{2}", " ", s)
    s = re.sub(_month_names, " ", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9% ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

# ----------------------------
# 📌 Normalize Table
# ----------------------------
def normalize_table_by_metric(df: pd.DataFrame, id_col=None):
    if id_col is None:
        id_col = df.columns[0]

    metric_cols = defaultdict(list)
    col_year_map = {}
    for col in df.columns:
        if col == id_col: 
            continue
        year = extract_year_from_text(col)
        metric = extract_metric_name(col)
        metric_cols[metric].append((col, year))
        col_year_map[col] = year

    enriched = pd.DataFrame()
    enriched[id_col] = df[id_col].astype(str)

    mapping = {}
    for metric, col_list in metric_cols.items():
        with_year = [(c, y) for c, y in col_list if y]
        if len(with_year) >= 2:
            sorted_by_year = sorted(with_year, key=lambda x: x[1], reverse=True)
            curr, prev = sorted_by_year[0][0], sorted_by_year[1][0]
        elif len(with_year) == 1:
            curr, prev = with_year[0][0], None
        else:
            continue

        mapping[metric] = {"current": curr, "previous": prev}
        safe_metric = re.sub(r"\s+", "_", metric.strip())
        is_percent_metric = "%" in metric or "share" in metric or "rate" in metric

        if curr: enriched[f"{safe_metric}_current"] = df[curr].apply(clean_numeric_cell)
        if prev: enriched[f"{safe_metric}_previous"] = df[prev].apply(clean_numeric_cell)

        if curr and prev:
            if is_percent_metric:
                enriched[f"{safe_metric}_yoy_pp"] = (
                    enriched[f"{safe_metric}_current"] - enriched[f"{safe_metric}_previous"]
                )
            else:
                enriched[f"{safe_metric}_yoy_pct"] = (
                    (enriched[f"{safe_metric}_current"] - enriched[f"{safe_metric}_previous"])
                    / enriched[f"{safe_metric}_previous"] * 100
                )

    return enriched, mapping

def format_with_units(val, col):
    if pd.isna(val) or val == "":
        return ""
    try:
        num = float(val)
    except Exception:
        return str(val)  # fallback: return as-is if it's text
    # Decide whether to show % or plain number
    if col.endswith("_pp") or col.endswith("_pct") or "share" in col or "rate" in col or "%" in col:
        return f"{num:.2f}%"
    return f"{num:.2f}"