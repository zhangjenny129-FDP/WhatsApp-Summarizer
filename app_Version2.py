import streamlit as st
import openai
from datetime import datetime, date
import re
import json
from typing import Optional, List, Tuple, Dict, Any
from dateutil import parser as dateparser
import time
import io
import csv

# --- APP CONFIGURATION ---
st.set_page_config(page_title="WhatsApp Summarizer", page_icon="📝", layout="wide")
st.title("📝 WhatsApp Group Summarizer")
st.markdown("Turn messy WhatsApp chat exports into clean meeting minutes, decisions, and actionable to-dos. Now with automatic chunking and improved date parsing for more locales.")

# --- SIDEBAR: Settings ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    model = st.selectbox("Model", options=["gpt-4", "gpt-4o", "gpt-3.5-turbo"], index=0)
    uploaded_file = st.file_uploader("Upload WhatsApp (.txt) export", type="txt")
    max_chars_per_request = st.slider("Max characters per model request (chunk size)", min_value=2000, max_value=120000, value=20000, step=1000)
    overlap_chars = st.slider("Chunk overlap (to preserve context)", min_value=0, max_value=2000, value=400, step=100)
    st.markdown("---")
    st.markdown(
        "Notes:\n\n"
        "- The app splits large chats into overlapping chunks so the model can process very large exports.\n"
        "- Date parsing uses dateutil to handle many locale formats, including long month names.\n"
        "- You can download results as JSON, Markdown, or CSV (generic, Notion-compatible, and Asana-compatible CSVs)."
    )

# --- HELPERS: Date parsing and JSON extraction ---
# Try to identify a date part at the start of a line commonly found in WhatsApp exports.
leading_date_part_regex = re.compile(r'^\[?(?P<datepart>[^-\n\r:]+?)(?:,?\s*\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM|am|pm)?)?\s*(?:-|–|—)\s*', re.IGNORECASE)

def parse_date_from_line(line: str) -> Optional[date]:
    """
    Attempts to parse a date from the beginning of a line using several strategies:
    - Extracts a leading date/time chunk (before the " - " that WhatsApp uses) and tries dateutil.parse
    - Tries direct regex for numeric dates
    Returns a datetime.date or None.
    """
    if not line or len(line.strip()) == 0:
        return None

    # 1) Try to capture the leading date/time portion before the " - " or " - Name"
    m = leading_date_part_regex.match(line.strip())
    candidates = []
    if m:
        candidates.append(m.group("datepart").strip())

    # 2) fallback: find explicit date-like substrings
    date_like = re.findall(r'(\d{1,4}[-/.\s]\d{1,2}[-/.\s]\d{1,4})', line)
    for d in date_like:
        candidates.append(d)

    # 3) also attempt to find month names
    month_like = re.findall(r'([A-Za-z]{3,9}\s+\d{1,2}(?:,?\s*\d{2,4})?)', line)
    for d in month_like:
        candidates.append(d)

    # Try parsing candidates with different dayfirst options
    for c in candidates:
        c_clean = c.replace('.', '/').replace('\\', '/')
        try:
            # First try with dayfirst=False (US-style), then dayfirst=True (European / HK sometimes)
            dt = dateparser.parse(c_clean, fuzzy=True, dayfirst=False)
            if dt:
                return dt.date()
        except Exception:
            pass
        try:
            dt = dateparser.parse(c_clean, fuzzy=True, dayfirst=True)
            if dt:
                return dt.date()
        except Exception:
            pass

    # Last fallback: try parsing the whole line
    try:
        dt = dateparser.parse(line, fuzzy=True, dayfirst=True)
        if dt:
            return dt.date()
    except Exception:
        pass

    return None

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Finds the first JSON object in text and returns a parsed dict, or None.
    """
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or start > end:
        return None
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        # Try to be more lenient: replace single quotes with double quotes (best-effort)
        try:
            candidate2 = candidate.replace("'", '"')
            return json.loads(candidate2)
        except Exception:
            return None

def merge_results(parsed_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple parsed JSON results from chunked responses.
    - meeting_minutes: unique list (preserve order)
    - decisions: unique list
    - todos: merge by task string (deduplicate), keep assignee/due_date if present
    """
    merged = {"meeting_minutes": [], "decisions": [], "todos": []}
    seen_minutes = set()
    seen_decisions = set()
    seen_todos = set()

    for p in parsed_list:
        if not isinstance(p, dict):
            continue
        for m in p.get("meeting_minutes", []) or []:
            key = m.strip()
            if key and key not in seen_minutes:
                merged["meeting_minutes"].append(m.strip())
                seen_minutes.add(key)
        for d in p.get("decisions", []) or []:
            key = d.strip()
            if key and key not in seen_decisions:
                merged["decisions"].append(d.strip())
                seen_decisions.add(key)
        for t in p.get("todos", []) or []:
            task = (t.get("task") or "").strip()
            if not task:
                continue
            if task not in seen_todos:
                # keep assignee/due_date if present
                merged["todos"].append({
                    "task": task,
                    "assignee": (t.get("assignee") or "").strip(),
                    "due_date": (t.get("due_date") or "").strip()
                })
                seen_todos.add(task)

    return merged

def build_markdown_from_parsed(parsed: Dict[str, Any]) -> str:
    md = ["# Meeting Summary\n"]
    if parsed.get("meeting_minutes"):
        md.append("## Meeting Minutes")
        for m in parsed["meeting_minutes"]:
            md.append(f"- {m}")
    if parsed.get("decisions"):
        md.append("\n## Decisions")
        for d in parsed["decisions"]:
            md.append(f"- {d}")
    if parsed.get("todos"):
        md.append("\n## To-dos")
        for t in parsed["todos"]:
            extra = []
            if t.get("assignee"): extra.append(f"assignee: {t['assignee']}")
            if t.get("due_date"): extra.append(f"due: {t['due_date']}")
            extras = f" ({', '.join(extra)})" if extra else ""
            md.append(f"- {t['task']}{extras}")
    return "\n".join(md)

def todos_to_csv_bytes(todos: List[Dict[str, str]], headers: List[str]) -> bytes:
    """
    Create a CSV (bytes) from todos with given headers order.
    headers should include column names that map to keys like 'task','assignee','due_date' or others.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for t in todos:
        row = {k: t.get(k, "") for k in headers}
        writer.writerow(row)
    return output.getvalue().encode("utf-8")

# --- MAIN LOGIC ---
if uploaded_file and api_key:
    # Read and decode file
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    # Detect dates from the file (scan lines)
    found_dates: List[date] = []
    for line in lines:
        d = parse_date_from_line(line)
        if d:
            found_dates.append(d)

    if not found_dates:
        st.error("Could not detect dates in the file. Please ensure it's a standard WhatsApp export or try a different file.")
    else:
        min_d, max_d = min(found_dates), max(found_dates)
        st.subheader("Select Time Frame")
        selected_range = st.date_input(
            "Select start and end dates:",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d
        )

        # Accept either a single date or a tuple of two dates
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        elif isinstance(selected_range, date):
            start_date = end_date = selected_range
        else:
            start_date = min_d
            end_date = max_d

        st.markdown(f"Filtering messages from **{start_date}** to **{end_date}**")

        if st.button("Generate Summary"):
            # Filter lines by date
            filtered_lines: List[str] = []
            for line in lines:
                d = parse_date_from_line(line)
                if d and start_date <= d <= end_date:
                    filtered_lines.append(line)
            if not filtered_lines:
                st.warning("No messages matched the selected date range.")
            else:
                filtered_text = "\n".join(filtered_lines)
                total_chars = len(filtered_text)
                st.info(f"Filtered content: {len(filtered_lines)} lines, ~{total_chars} characters.")

                # Chunking: split by character length with overlap
                chunks: List[str] = []
                if total_chars <= max_chars_per_request:
                    chunks = [filtered_text]
                else:
                    start = 0
                    while start < total_chars:
                        end = min(start + max_chars_per_request, total_chars)
                        chunk = filtered_text[start:end]
                        chunks.append(chunk)
                        if end == total_chars:
                            break
                        # step forward by chunk size - overlap
                        start = max(0, end - overlap_chars)

                st.write(f"Prepared {len(chunks)} chunk(s) to send to the model.")

                # Prepare prompts
                system_prompt = (
                    "You are an assistant that extracts structured meeting minutes, decisions, and todos from WhatsApp chat text. "
                    "Return a JSON object with keys: 'meeting_minutes' (array of short bullet strings), "
                    "'decisions' (array of short strings), and 'todos' (array of objects with keys 'task', 'assignee' (optional), 'due_date' (optional)). "
                    "Keep the JSON strict and machine-parseable and place it at the start of the response. After the JSON you may include a brief human-readable summary."
                )

                openai.api_key = api_key
                parsed_outputs: List[Dict[str, Any]] = []
                raw_outputs: List[str] = []

                # Call the model for each chunk, with a small pause between requests to be polite
                for i, chunk in enumerate(chunks, start=1):
                    user_prompt = (
                        f"Chunk {i}/{len(chunks)} of the WhatsApp export. Extract meeting minutes, decisions, and todos.\n\n"
                        "Raw chat chunk:\n\n" + chunk + "\n\n"
                        "Return: a JSON object as specified (meeting_minutes, decisions, todos) at the top of the response, then a short markdown summary."
                    )
                    with st.spinner(f"Analyzing chunk {i}/{len(chunks)}..."):
                        try:
                            resp = openai.ChatCompletion.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.1,
                                max_tokens=1200,
                            )
                            model_output = resp["choices"][0]["message"]["content"]
                        except Exception as e:
                            st.error(f"OpenAI request failed on chunk {i}: {e}")
                            model_output = ""
                        raw_outputs.append(model_output)
                        parsed = extract_json_from_text(model_output)
                        if parsed:
                            parsed_outputs.append(parsed)
                        else:
                            # If parsing failed, add an empty structure with the raw text in a 'notes' field
                            parsed_outputs.append({"meeting_minutes": [], "decisions": [], "todos": [], "notes": model_output})

                    # polite pause (simple rate-limit avoidance)
                    time.sleep(0.5)

                # Merge parsed results
                merged = merge_results(parsed_outputs)
                st.success("Analysis Complete — merged results from chunks.")

                # Display merged JSON
                st.subheader("Merged Extracted JSON")
                st.json(merged)

                # Human-readable markdown summary
                markdown_content = build_markdown_from_parsed(merged)
                st.subheader("Human-readable Summary")
                st.markdown(markdown_content)

                # Provide downloads: JSON, Markdown, CSV (generic), Notion-compatible CSV, Asana-compatible CSV
                st.download_button("Download JSON", data=json.dumps(merged, indent=2), file_name="whatsapp_summary.json", mime="application/json")
                st.download_button("Download Markdown", data=markdown_content, file_name="meeting_summary.md", mime="text/markdown")

                # Generic CSV for todos
                todos = merged.get("todos", [])
                if todos:
                    generic_headers = ["task", "assignee", "due_date"]
                    csv_bytes = todos_to_csv_bytes(todos, generic_headers)
                    st.download_button("Download Todos CSV (generic)", data=csv_bytes, file_name="todos_generic.csv", mime="text/csv")

                    # Notion-compatible CSV: Notion imports CSV with a "Name" column and additional columns; we'll map:
                    notion_headers = ["Name", "Assignee", "Due Date", "Notes"]
                    notion_rows = []
                    for t in todos:
                        notion_rows.append({
                            "Name": t.get("task", ""),
                            "Assignee": t.get("assignee", ""),
                            "Due Date": t.get("due_date", ""),
                            "Notes": ""
                        })
                    # produce CSV
                    notion_csv_io = io.StringIO()
                    writer = csv.DictWriter(notion_csv_io, fieldnames=notion_headers)
                    writer.writeheader()
                    for r in notion_rows:
                        writer.writerow(r)
                    st.download_button("Download Todos CSV (Notion-friendly)", data=notion_csv_io.getvalue().encode("utf-8"), file_name="todos_notion.csv", mime="text/csv")

                    # Asana-compatible CSV: Asana expects columns like "Name,Notes,Assignee,Projects,Due Date"
                    asana_headers = ["Name", "Notes", "Assignee", "Projects", "Due Date"]
                    asana_rows = []
                    for t in todos:
                        asana_rows.append({
                            "Name": t.get("task", ""),
                            "Notes": "",
                            "Assignee": t.get("assignee", ""),
                            "Projects": "",
                            "Due Date": t.get("due_date", "")
                        })
                    asana_csv_io = io.StringIO()
                    writer = csv.DictWriter(asana_csv_io, fieldnames=asana_headers)
                    writer.writeheader()
                    for r in asana_rows:
                        writer.writerow(r)
                    st.download_button("Download Todos CSV (Asana-friendly)", data=asana_csv_io.getvalue().encode("utf-8"), file_name="todos_asana.csv", mime="text/csv")
                else:
                    st.info("No todos extracted.")

                # Show raw outputs per chunk (expandable)
                exp = st.expander("Show raw model outputs per chunk")
                for i, out in enumerate(raw_outputs, start=1):
                    exp.subheader(f"Chunk {i} output")
                    exp.text_area(f"chunk_{i}_raw", value=out or "<no output>", height=220)

else:
    st.info("Please enter your API Key and upload a .txt file in the sidebar to begin.")