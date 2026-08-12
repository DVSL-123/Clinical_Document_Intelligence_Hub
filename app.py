import os
import json
import base64
import streamlit as st
from dotenv import load_dotenv
from google import genai

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
    st.error("❌ GEMINI_API_KEY is not configured.")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.6-flash"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical Document Intelligence Hub",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Clinical Document Intelligence Hub")
st.write("Upload clinical documents and images to extract structured information and actionable intelligence.")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📋 Analysis Pipeline")
    st.write("📄 Upload Document")
    st.write("🔍 Extract Information")
    st.write("🤖 Gemini AI Analysis")
    st.write("📊 Structured Output")
    st.write("⚠️ Risk Identification")
    st.write("📋 Recommended Next Step")
    st.divider()
    st.write("**AI Model:**")
    st.write(MODEL)

# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📄 Upload Clinical Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, TXT, PNG, JPG or JPEG files",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ============================================================
# PROMPT WITH SOURCE TRACEABILITY
# ============================================================

prompt = """
You are a Clinical Document Intelligence assistant.

Analyze the uploaded clinical document or image.

Return the information in JSON with both the extracted value and the source snippet.

If information is missing, write "Not available".

JSON structure:

{
    "patient_summary": {
        "name": {"value": "string", "source": "string"},
        "age": {"value": "string", "source": "string"},
        "gender": {"value": "string", "source": "string"},
        "summary": {"value": "string", "source": "string"}
    },
    "diagnoses": [{"value": "string", "source": "string"}],
    "medications": [{"value": "string", "source": "string"}],
    "allergies": [{"value": "string", "source": "string"}],
    "laboratory_results": [{"value": "string", "source": "string"}],
    "clinical_findings": [{"value": "string", "source": "string"}],
    "risk_flags": [{
        "severity": "High/Moderate/Low",
        "issue": "string",
        "reason": "string",
        "source": "string"
    }],
    "recommended_next_step": {"value": "string", "source": "string"},
    "confidence": {"value": "High/Moderate/Low", "source": "string"}
}

Important:
1. Never invent patient information.
2. Use only information found in the document.
3. Include the source snippet (page, paragraph, or text line).
"""

# ============================================================
# ANALYSIS FUNCTION
# ============================================================

@st.cache_data
def analyze_with_gemini(file_bytes, file_type):
    encoded_file = base64.b64encode(file_bytes).decode("utf-8")
    inputs = []

    if file_type == "application/pdf":
        inputs.append({"type": "document", "data": encoded_file, "mime_type": "application/pdf"})
    elif file_type in ["image/png", "image/jpeg"]:
        inputs.append({"type": "image", "data": encoded_file, "mime_type": file_type})
    elif file_type == "text/plain":
        text = file_bytes.decode("utf-8", errors="ignore")
        inputs.append({"type": "text", "text": text})

    # Add instructions last
    inputs.append({"type": "text", "text": prompt})

    interaction = client.interactions.create(
        model=MODEL,
        input=inputs,
        response_format={"type": "text", "mime_type": "application/json"}
    )
    return json.loads(interaction.output_text)

# ============================================================
# REAL-TIME ANALYSIS
# ============================================================

results = {}
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} document(s) uploaded.")
    for file in uploaded_files:
        st.write(f"📄 **{file.name}** — {file.type} — {file.size / 1024:.1f} KB")
        try:
            result = analyze_with_gemini(file.getvalue(), file.type)
            results[file.name] = result
            st.success(f"✅ {file.name} analyzed in real time")
        except Exception as e:
            st.error(f"❌ Error analyzing {file.name}")
            st.code(str(e))

# ============================================================
# DISPLAY RESULTS
# ============================================================

if results:
    st.divider()
    st.header("📊 Clinical Intelligence")

    for filename, result in results.items():
        st.subheader(f"📄 Results for {filename}")
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Patient Summary", "🧪 Clinical Findings", "⚠️ Risk Flags", "📋 Recommendation"])

        # TAB 1 — PATIENT SUMMARY
        with tab1:
            patient = result["patient_summary"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Patient", patient["name"]["value"])
            with col2:
                st.metric("Age", patient["age"]["value"])
            with col3:
                st.metric("Gender", patient["gender"]["value"])

            st.subheader("Patient Summary")
            st.write(patient["summary"]["value"])
            with st.expander("Source"):
                st.write(patient["summary"]["source"])

            st.subheader("Diagnoses")
            if result["diagnoses"]:
                for diagnosis in result["diagnoses"]:
                    st.write(f"• {diagnosis['value']}")
                    with st.expander("Source"):
                        st.write(diagnosis["source"])
            else:
                st.write("No diagnoses identified.")

            st.subheader("Medications")
            if result["medications"]:
                for medication in result["medications"]:
                    st.write(f"• {medication['value']}")
                    with st.expander("Source"):
                        st.write(medication["source"])
            else:
                st.write("No medications identified.")

            st.subheader("Allergies")
            if result["allergies"]:
                for allergy in result["allergies"]:
                    st.write(f"• {allergy['value']}")
                    with st.expander("Source"):
                        st.write(allergy["source"])
            else:
                st.write("No allergies identified.")

        # TAB 2 — CLINICAL FINDINGS
        with tab2:
            st.subheader("🧪 Laboratory Results")
            if result["laboratory_results"]:
                for lab in result["laboratory_results"]:
                    st.write(f"• {lab['value']}")
                    with st.expander("Source"):
                        st.write(lab["source"])
            else:
                st.write("No laboratory results identified.")

            st.subheader("🔎 Clinical Findings")
            if result["clinical_findings"]:
                for finding in result["clinical_findings"]:
                    st.write(f"• {finding['value']}")
                    with st.expander("Source"):
                        st.write(finding["source"])
            else:
                st.write("No significant clinical findings identified.")

        # TAB 3 — RISK FLAGS
        with tab3:
            risks = result["risk_flags"]
            if risks:
                for risk in risks:
                    severity = risk["severity"].lower()
                    issue = risk["issue"]
                    reason = risk["reason"]
                    source = risk["source"]

                    if severity == "high":
                        st.error(f"🔴 HIGH — {issue}\n\n{reason}")
                    elif severity == "moderate":
                        st.warning(f"🟠 MODERATE — {issue}\n\n{reason}")
                    else:
                        st.info(f"🟡 {risk['severity'].upper()} — {issue}\n\n{reason}")

                    with st.expander("Source"):
                        st.write(source)
            else:
                st.success("✅ No potential risk flags identified.")

        # TAB 4 — RECOMMENDATION
        with tab4:
            st.subheader("📋 Recommended Next Step")
            st.info(result["recommended_next_step"]["value"])
            with st.expander("Source"):
                st.write(result["recommended_next_step"]["source"])

            st.subheader("AI Confidence")
            st.write(result["confidence"]["value"])
            with st.expander("Source"):
                st.write(result["confidence"]["source"])

        # RAW JSON
        st.divider()
        with st.expander("🔍 View Structured JSON"):
            st.json(result)

# ============================================================
# DISCLAIMER
# ============================================================

st.divider()
st.caption(
    "⚠️ POC only. This application provides AI-assisted "
    "information extraction and decision support. "
    "It does not replace professional")