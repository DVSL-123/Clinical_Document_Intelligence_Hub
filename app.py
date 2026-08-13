import os
import json
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# SETUP
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = None

if not API_KEY:
    st.error(
        "❌ GEMINI_API_KEY is not configured. "
        "Set it in a .env file or in Streamlit secrets."
    )
    st.stop()

# Gemini API client
client = genai.Client(api_key=API_KEY)

# ============================================================
# GEMINI 3.6 FLASH
# ============================================================

MODEL = "gemini-3.6-flash"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical Document Intelligence Hub",
    page_icon="🏥",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "consolidated_result" not in st.session_state:
    st.session_state.consolidated_result = None

if "processed_file_keys" not in st.session_state:
    st.session_state.processed_file_keys = set()

if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []


# ============================================================
# SMALL HELPERS
# ============================================================

def show_value(value, source, bullet=False):
    value = value if value else "Not available"
    prefix = "• " if bullet else ""

    row = st.columns([12, 1])

    with row[0]:
        st.write(f"{prefix}{value}")

    with row[1]:
        with st.popover("ℹ️"):
            st.caption("**Source**")
            st.write(source if source else "Source not available.")


def show_list(items, empty_message):
    if items:
        for item in items:
            show_value(
                item.get("value", "Not available"),
                item.get("source", ""),
                bullet=True
            )
    else:
        st.write(empty_message)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🏥 Clinical Document Intelligence Hub")

st.write(
    "Upload clinical documents for a patient. Each new document is "
    "analyzed independently and merged into the existing summary — "
    "nothing already known gets thrown away or rewritten."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.caption("🔧 Build: Gemini-3.6-flash")

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    with st.expander(
        "🔬 Check available models for your API key",
        expanded=True
    ):

        if st.button("Run check"):

            try:

                available = []

                for m in client.models.list():

                    model_name = getattr(m, "name", "")

                    # Current Gemini API model objects may expose
                    # supported actions differently depending on SDK version.
                    supported_actions = (
                        getattr(m, "supported_actions", None)
                        or getattr(m, "supported_generation_methods", None)
                        or []
                    )

                    if (
                        "generateContent" in supported_actions
                        or "generate_content" in supported_actions
                    ):
                        available.append(model_name)

                if available:

                    st.success(
                        f"Found {len(available)} model(s) "
                        "that support content generation:"
                    )

                    for name in available:
                        st.code(name)

                else:

                    st.warning(
                        "No models supporting generateContent were returned "
                        "for this API key."
                    )

            except Exception as e:

                st.error(
                    "Could not list models. This usually means the API key "
                    "is invalid, expired, restricted, or the Gemini API "
                    "is not enabled."
                )

                st.code(str(e))

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.header("📋 Analysis Pipeline")

    st.write("📄 Upload Document")
    st.write("🔍 Extract Information")
    st.write("🤖 Gemini 3.6 Flash Analysis")
    st.write("🔄 Merge Into Existing Summary")
    st.write("⚠️ Risk Identification")
    st.write("📋 Recommended Next Step")

    st.divider()

    st.write("**AI Model:**")
    st.code(MODEL)

    st.divider()

    if st.button("🗑️ Start over (clear summary)"):

        st.session_state.consolidated_result = None
        st.session_state.processed_file_keys = set()
        st.session_state.processed_file_names = []

        st.rerun()


# ============================================================
# PROMPT
# ============================================================

SINGLE_DOC_PROMPT = """
You are a Clinical Document Intelligence assistant.

Analyze ONLY the uploaded clinical document.

Your job is to extract structured clinical information from THIS
document and return it as valid JSON.

Do not use outside knowledge to invent patient-specific information.

Return exactly this JSON structure:

{
  "patient_summary": {
    "name": {
      "value": "string",
      "source": "string"
    },
    "age": {
      "value": "string",
      "source": "string"
    },
    "gender": {
      "value": "string",
      "source": "string"
    },
    "summary": {
      "value": "string",
      "source": "string"
    }
  },

  "diagnoses": [
    {
      "value": "string",
      "source": "string"
    }
  ],

  "medications": [
    {
      "value": "string",
      "source": "string"
    }
  ],

  "allergies": [
    {
      "value": "string",
      "source": "string"
    }
  ],

  "laboratory_results": [
    {
      "value": "string",
      "source": "string"
    }
  ],

  "clinical_findings": [
    {
      "value": "string",
      "source": "string"
    }
  ],

  "risk_flags": [
    {
      "severity": "High/Moderate/Low",
      "issue": "string",
      "reason": "string",
      "source": "string"
    }
  ],

  "recommended_next_step": {
    "value": "string",
    "source": "string"
  },

  "confidence": {
    "value": "High/Moderate/Low",
    "source": "string"
  }
}

IMPORTANT RULES:

1. Never invent patient information.

2. Use ONLY information present in THIS document.

3. If a field is not present, use:
   "Not available"

4. Do not infer a diagnosis that is not explicitly supported
   by the document.

5. Risk flags must be based ONLY on evidence present in
   this document.

6. For every extracted item, provide a source describing
   where it was found.

7. For PDFs, identify the page number whenever possible.

8. For tables, identify the relevant table or row whenever possible.

9. For images/scanned documents, identify the visible section
   or text containing the information.

10. Do not combine information from other documents.

11. Return valid JSON only.

12. Do not put markdown fences around the JSON.

13. Keep the extracted information concise but complete.

14. The recommended next step is an AI-assisted suggestion based
    strictly on the clinical information present in this document.
"""


# ============================================================
# FILE BUILDER
# ============================================================

def _build_file_part(
    file_bytes: bytes,
    file_type: str,
    file_name: str
):

    if file_type == "application/pdf":

        return types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf"
        )

    elif file_type in ("image/png", "image/jpeg"):

        return types.Part.from_bytes(
            data=file_bytes,
            mime_type=file_type
        )

    elif file_type == "text/plain":

        text = file_bytes.decode(
            "utf-8",
            errors="ignore"
        )

        return types.Part.from_text(
            text=f"[Document: {file_name}]\n{text}"
        )

    else:

        raise ValueError(
            f"Unsupported file type: {file_type}"
        )


# ============================================================
# GEMINI 3.6 FLASH — SINGLE DOCUMENT ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def analyze_single_file(
    file_bytes: bytes,
    file_type: str,
    file_name: str
):

    file_part = _build_file_part(
        file_bytes,
        file_type,
        file_name
    )

    prompt_part = types.Part.from_text(
        text=SINGLE_DOC_PROMPT
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    file_part,
                    prompt_part
                ]
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    try:

        return json.loads(response.text)

    except json.JSONDecodeError as e:

        raise ValueError(
            "Gemini returned a response that was not valid JSON.\n\n"
            f"Raw response:\n{response.text}"
        ) from e


# ============================================================
# MERGE HELPERS
# ============================================================

EMPTY_VALUES = {
    None,
    "",
    "not available",
    "n/a"
}


def _is_empty(value):

    if value is None:
        return True

    return str(value).strip().lower() in EMPTY_VALUES


# ============================================================
# SINGLE VALUE MERGE
# ============================================================

def _merge_single_field(
    existing,
    new,
    file_name
):

    if existing is None:
        existing = {
            "value": "Not available",
            "source": ""
        }

    if new is None:
        new = {
            "value": "Not available",
            "source": ""
        }

    if (
        _is_empty(existing.get("value"))
        and not _is_empty(new.get("value"))
    ):

        return {
            "value": new["value"],
            "source": new.get(
                "source",
                f"From {file_name}"
            )
        }

    return existing


# ============================================================
# SUMMARY MERGE
# ============================================================

def _merge_summary_text(
    existing,
    new,
    file_name
):

    if existing is None:
        existing = {
            "value": "Not available",
            "source": ""
        }

    if new is None:
        new = {
            "value": "Not available",
            "source": ""
        }

    if _is_empty(existing.get("value")):

        return {
            "value": new.get(
                "value",
                "Not available"
            ),
            "source": new.get(
                "source",
                f"From {file_name}"
            )
        }

    if _is_empty(new.get("value")):

        return existing

    if (
        str(new["value"]).strip().lower()
        == str(existing["value"]).strip().lower()
    ):

        return existing

    combined_value = (
        f'{existing["value"]} '
        f'{new["value"]}'
    )

    combined_source = (
        f'{existing.get("source", "")}; '
        f'{file_name}: '
        f'{new.get("source", "")}'
    )

    return {
        "value": combined_value,
        "source": combined_source
    }


# ============================================================
# LIST MERGE
# ============================================================

def _merge_item_list(
    existing_list,
    new_list,
    file_name,
    dedup_key="value"
):

    existing_list = list(existing_list or [])

    seen = {
        str(item.get(dedup_key, ""))
        .strip()
        .lower()
        for item in existing_list
    }

    for item in (new_list or []):

        key = (
            str(item.get(dedup_key, ""))
            .strip()
            .lower()
        )

        if not key:
            continue

        if key not in seen:

            if not item.get("source"):

                item["source"] = (
                    f"From {file_name}"
                )

            existing_list.append(item)
            seen.add(key)

    return existing_list


# ============================================================
# RISK FLAG MERGE
# ============================================================

def _merge_risk_flags(
    existing_list,
    new_list,
    file_name
):

    existing_list = list(existing_list or [])

    seen = {
        str(item.get("issue", ""))
        .strip()
        .lower()
        for item in existing_list
    }

    for item in (new_list or []):

        key = (
            str(item.get("issue", ""))
            .strip()
            .lower()
        )

        if not key:
            continue

        if key not in seen:

            if not item.get("source"):

                item["source"] = (
                    f"From {file_name}"
                )

            existing_list.append(item)
            seen.add(key)

    return existing_list


# ============================================================
# CONFIDENCE
# ============================================================

CONFIDENCE_RANK = {
    "low": 1,
    "moderate": 2,
    "high": 3
}


# ============================================================
# MAIN MERGE
# ============================================================

def merge_new_extraction(
    existing,
    new,
    file_name
):

    # --------------------------------------------------------
    # FIRST DOCUMENT
    # --------------------------------------------------------

    if existing is None:

        for item in (
            new.get("diagnoses", [])
            + new.get("medications", [])
            + new.get("allergies", [])
            + new.get("laboratory_results", [])
            + new.get("clinical_findings", [])
            + new.get("risk_flags", [])
        ):

            if not item.get("source"):

                item["source"] = (
                    f"From {file_name}"
                )

        return new

    # --------------------------------------------------------
    # EXISTING PATIENT DATA
    # --------------------------------------------------------

    merged = {}

    existing_patient = existing.get(
        "patient_summary",
        {}
    )

    new_patient = new.get(
        "patient_summary",
        {}
    )

    merged["patient_summary"] = {

        "name": _merge_single_field(
            existing_patient.get("name"),
            new_patient.get("name"),
            file_name
        ),

        "age": _merge_single_field(
            existing_patient.get("age"),
            new_patient.get("age"),
            file_name
        ),

        "gender": _merge_single_field(
            existing_patient.get("gender"),
            new_patient.get("gender"),
            file_name
        ),

        "summary": _merge_summary_text(
            existing_patient.get("summary"),
            new_patient.get("summary"),
            file_name
        )
    }

    # --------------------------------------------------------
    # LISTS
    # --------------------------------------------------------

    merged["diagnoses"] = _merge_item_list(
        existing.get("diagnoses"),
        new.get("diagnoses"),
        file_name
    )

    merged["medications"] = _merge_item_list(
        existing.get("medications"),
        new.get("medications"),
        file_name
    )

    merged["allergies"] = _merge_item_list(
        existing.get("allergies"),
        new.get("allergies"),
        file_name
    )

    merged["laboratory_results"] = _merge_item_list(
        existing.get("laboratory_results"),
        new.get("laboratory_results"),
        file_name
    )

    merged["clinical_findings"] = _merge_item_list(
        existing.get("clinical_findings"),
        new.get("clinical_findings"),
        file_name
    )

    merged["risk_flags"] = _merge_risk_flags(
        existing.get("risk_flags"),
        new.get("risk_flags"),
        file_name
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    existing_rec = existing.get(
        "recommended_next_step",
        {}
    )

    new_rec = new.get(
        "recommended_next_step",
        {}
    )

    if _is_empty(
        existing_rec.get("value")
    ):

        merged["recommended_next_step"] = new_rec

    elif (
        not _is_empty(new_rec.get("value"))
        and
        str(new_rec["value"]).strip().lower()
        != str(
            existing_rec.get("value", "")
        ).strip().lower()
    ):

        merged["recommended_next_step"] = {

            "value": (
                f'{existing_rec["value"]} '
                f'Also, based on {file_name}: '
                f'{new_rec["value"]}'
            ),

            "source": (
                f'{existing_rec.get("source", "")}; '
                f'{file_name}: '
                f'{new_rec.get("source", "")}'
            )
        }

    else:

        merged["recommended_next_step"] = existing_rec

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    existing_conf = existing.get(
        "confidence",
        {
            "value": "Not available",
            "source": ""
        }
    )

    new_conf = new.get(
        "confidence",
        {
            "value": "Not available",
            "source": ""
        }
    )

    existing_rank = CONFIDENCE_RANK.get(
        str(
            existing_conf.get("value", "")
        ).lower(),
        0
    )

    new_rank = CONFIDENCE_RANK.get(
        str(
            new_conf.get("value", "")
        ).lower(),
        0
    )

    if new_rank and (
        existing_rank == 0
        or new_rank < existing_rank
    ):

        merged["confidence"] = new_conf

    elif existing_rank:

        merged["confidence"] = existing_conf

    else:

        merged["confidence"] = new_conf

    return merged


# ============================================================
# LAYOUT
# ============================================================

col_input, col_results = st.columns(
    [1, 2],
    gap="large"
)

analysis_error = None


# ============================================================
# LEFT — UPLOAD
# ============================================================

with col_input:

    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, PNG, JPG or JPEG files (same patient)",
        type=[
            "pdf",
            "txt",
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        with st.expander(
            "🐛 Debug: what the app currently sees",
            expanded=True
        ):

            st.write(
                f"`uploaded_files` count this run: "
                f"**{len(uploaded_files)}**"
            )

            for f in uploaded_files:

                key = (
                    f.name,
                    f.size
                )

                already_done = (
                    key
                    in st.session_state.processed_file_keys
                )

                st.write(
                    f"- `{f.name}` "
                    f"({f.size} bytes) — "
                    f"{'already processed, will SKIP' if already_done else 'NEW, will analyze'}"
                )

            st.write(
                "`processed_file_keys` in session so far:"
            )

            st.write(
                st.session_state.processed_file_keys
            )

        # ----------------------------------------------------
        # FIND NEW FILES
        # ----------------------------------------------------

        new_files = [

            f
            for f in uploaded_files

            if (
                f.name,
                f.size
            )
            not in st.session_state.processed_file_keys

        ]

        # ----------------------------------------------------
        # PROCESS NEW FILES
        # ----------------------------------------------------

        if new_files:

            for f in new_files:

                try:

                    with st.spinner(
                        f"Analyzing {f.name} with Gemini 3.6 Flash..."
                    ):

                        extraction = analyze_single_file(
                            f.getvalue(),
                            f.type,
                            f.name
                        )

                    st.session_state.consolidated_result = (
                        merge_new_extraction(
                            st.session_state.consolidated_result,
                            extraction,
                            f.name
                        )
                    )

                    st.session_state.processed_file_keys.add(
                        (
                            f.name,
                            f.size
                        )
                    )

                    st.session_state.processed_file_names.append(
                        f.name
                    )

                    st.success(
                        f"✅ {f.name} merged into summary"
                    )

                except Exception as e:

                    analysis_error = str(e)

                    st.error(
                        f"❌ Error analyzing {f.name}"
                    )

                    st.code(
                        analysis_error
                    )

        # ----------------------------------------------------
        # INCLUDED DOCUMENTS
        # ----------------------------------------------------

        st.divider()

        st.write(
            "**Documents included in summary so far:**"
        )

        if st.session_state.processed_file_names:

            for name in (
                st.session_state.processed_file_names
            ):

                st.write(
                    f"✅ {name}"
                )

        else:

            st.write(
                "None yet."
            )

    else:

        st.info(
            "Upload one or more documents to begin analysis."
        )


# ============================================================
# RIGHT — RESULTS
# ============================================================

with col_results:

    st.header(
        "📊 Consolidated Clinical Intelligence"
    )

    result = (
        st.session_state.consolidated_result
    )

    if not result:

        st.info(
            "The consolidated summary will appear here "
            "once you upload a document on the left."
        )

    else:

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "👤 Patient Summary",
                "🧪 Clinical Findings",
                "⚠️ Risk Flags",
                "📋 Recommendation"
            ]
        )

        # ====================================================
        # TAB 1 — PATIENT SUMMARY
        # ====================================================

        with tab1:

            patient = result.get(
                "patient_summary",
                {}
            )

            c1, c2, c3 = st.columns(3)

            # ------------------------------------------------
            # NAME
            # ------------------------------------------------

            with c1:

                st.metric(
                    "Patient",
                    patient.get(
                        "name",
                        {}
                    ).get(
                        "value",
                        "Not available"
                    )
                )

                with st.popover("ℹ️ source"):

                    st.write(
                        patient.get(
                            "name",
                            {}
                        ).get(
                            "source"
                        )
                        or
                        "Source not available."
                    )

            # ------------------------------------------------
            # AGE
            # ------------------------------------------------

            with c2:

                st.metric(
                    "Age",
                    patient.get(
                        "age",
                        {}
                    ).get(
                        "value",
                        "Not available"
                    )
                )

                with st.popover("ℹ️ source"):

                    st.write(
                        patient.get(
                            "age",
                            {}
                        ).get(
                            "source"
                        )
                        or
                        "Source not available."
                    )

            # ------------------------------------------------
            # GENDER
            # ------------------------------------------------

            with c3:

                st.metric(
                    "Gender",
                    patient.get(
                        "gender",
                        {}
                    ).get(
                        "value",
                        "Not available"
                    )
                )

                with st.popover("ℹ️ source"):

                    st.write(
                        patient.get(
                            "gender",
                            {}
                        ).get(
                            "source"
                        )
                        or
                        "Source not available."
                    )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "Patient Summary"
            )

            summary = patient.get(
                "summary",
                {}
            )

            show_value(
                summary.get(
                    "value",
                    "Not available"
                ),
                summary.get(
                    "source",
                    ""
                )
            )

            # ------------------------------------------------
            # DIAGNOSES
            # ------------------------------------------------

            st.subheader(
                "Diagnoses"
            )

            show_list(
                result.get(
                    "diagnoses",
                    []
                ),
                "No diagnoses identified."
            )

            # ------------------------------------------------
            # MEDICATIONS
            # ------------------------------------------------

            st.subheader(
                "Medications"
            )

            show_list(
                result.get(
                    "medications",
                    []
                ),
                "No medications identified."
            )

            # ------------------------------------------------
            # ALLERGIES
            # ------------------------------------------------

            st.subheader(
                "Allergies"
            )

            show_list(
                result.get(
                    "allergies",
                    []
                ),
                "No allergies identified."
            )

        # ====================================================
        # TAB 2 — CLINICAL FINDINGS
        # ====================================================

        with tab2:

            st.subheader(
                "🧪 Laboratory Results"
            )

            show_list(
                result.get(
                    "laboratory_results",
                    []
                ),
                "No laboratory results identified."
            )

            st.subheader(
                "🔎 Clinical Findings"
            )

            show_list(
                result.get(
                    "clinical_findings",
                    []
                ),
                "No significant clinical findings identified."
            )

        # ====================================================
        # TAB 3 — RISK FLAGS
        # ====================================================

        with tab3:

            risks = result.get(
                "risk_flags",
                []
            )

            if risks:

                for risk in risks:

                    severity = risk.get(
                        "severity",
                        "Low"
                    )

                    issue = risk.get(
                        "issue",
                        "Potential risk"
                    )

                    reason = risk.get(
                        "reason",
                        "Reason not available"
                    )

                    source = risk.get(
                        "source",
                        ""
                    )

                    if severity.lower() == "high":

                        st.error(
                            f"🔴 HIGH — {issue}\n\n"
                            f"{reason}"
                        )

                    elif severity.lower() == "moderate":

                        st.warning(
                            f"🟠 MODERATE — {issue}\n\n"
                            f"{reason}"
                        )

                    else:

                        st.info(
                            f"🟡 {severity.upper()} — "
                            f"{issue}\n\n"
                            f"{reason}"
                        )

                    with st.popover(
                        "ℹ️ source"
                    ):

                        st.write(
                            source
                            or
                            "Source not available."
                        )

                    st.write("")

            else:

                st.success(
                    "✅ No potential risk flags identified."
                )

        # ====================================================
        # TAB 4 — RECOMMENDATION
        # ====================================================

        with tab4:

            st.subheader(
                "📋 Recommended Next Step"
            )

            recommendation = result.get(
                "recommended_next_step",
                {}
            )

            show_value(
                recommendation.get(
                    "value",
                    "Not available"
                ),
                recommendation.get(
                    "source",
                    ""
                )
            )

            st.subheader(
                "AI Confidence"
            )

            confidence = result.get(
                "confidence",
                {}
            )

            confidence_value = confidence.get(
                "value",
                "Not available"
            )

            confidence_source = confidence.get(
                "source",
                ""
            )

            if confidence_value.lower() == "high":

                st.success(
                    f"Confidence: {confidence_value}"
                )

            elif confidence_value.lower() == "moderate":

                st.warning(
                    f"Confidence: {confidence_value}"
                )

            else:

                st.info(
                    f"Confidence: {confidence_value}"
                )

            with st.popover("ℹ️ source"):

                st.write(
                    confidence_source
                    or
                    "Source not available."
                )

        # ====================================================
        # RAW JSON
        # ====================================================

        st.divider()

        with st.expander(
            "🔍 View Structured JSON"
        ):

            st.json(result)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ POC only. This application provides AI-assisted "
    "information extraction and decision-support insights. "
    "It does not replace professional clinical judgment."
)