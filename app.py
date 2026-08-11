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

# Current Gemini model
MODEL = "gemini-3.6-flash"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical Document Intelligence Hub",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Clinical Document Intelligence Hub")

st.write(
    "Upload clinical documents and images to extract "
    "structured information and actionable intelligence."
)


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
    type=[
        "pdf",
        "txt",
        "png",
        "jpg",
        "jpeg"
    ],
    accept_multiple_files=True
)


# ============================================================
# SHOW UPLOADED FILES
# ============================================================

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} document(s) uploaded."
    )

    for file in uploaded_files:

        st.write(
            f"📄 **{file.name}** "
            f"— {file.type} "
            f"— {file.size / 1024:.1f} KB"
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_files:

    analyze = st.button(
        "🔍 Analyze Documents",
        type="primary"
    )

    if analyze:

        try:

            # ------------------------------------------------
            # PROMPT
            # ------------------------------------------------

            prompt = """
You are a Clinical Document Intelligence assistant.

Analyze the uploaded clinical document or image.

The purpose of this system is to help clinical and
administrative teams understand unstructured documents.

DO NOT diagnose the patient.

Extract only information supported by the document.

Return the information in the exact JSON structure
provided below.

If information is missing, write "Not available".

For risk flags, identify potentially concerning
information that may require professional review.

Recommendations must be framed as possible next
steps for professional review and must NOT be
presented as medical diagnosis or treatment.

JSON structure:

{
    "patient_summary": {
        "name": "string",
        "age": "string",
        "gender": "string",
        "summary": "string"
    },

    "diagnoses": [
        "string"
    ],

    "medications": [
        "string"
    ],

    "allergies": [
        "string"
    ],

    "laboratory_results": [
        "string"
    ],

    "clinical_findings": [
        "string"
    ],

    "risk_flags": [
        {
            "severity": "High/Moderate/Low",
            "issue": "string",
            "reason": "string"
        }
    ],

    "recommended_next_step": "string",

    "confidence": "High/Moderate/Low"
}

Important:

1. Never invent patient information.
2. Use only information found in the document.
3. Do not make a definitive diagnosis.
4. Clearly identify abnormal or concerning findings.
5. Keep the summary concise.
6. Make the output useful for a healthcare team.
"""


            # ------------------------------------------------
            # CREATE GEMINI INPUT
            # ------------------------------------------------

            inputs = []

            for file in uploaded_files:

                file_bytes = file.getvalue()

                encoded_file = base64.b64encode(
                    file_bytes
                ).decode("utf-8")


                # PDF
                if file.type == "application/pdf":

                    inputs.append(
                        {
                            "type": "document",
                            "data": encoded_file,
                            "mime_type": "application/pdf"
                        }
                    )


                # IMAGE
                elif file.type in [
                    "image/png",
                    "image/jpeg"
                ]:

                    inputs.append(
                        {
                            "type": "image",
                            "data": encoded_file,
                            "mime_type": file.type
                        }
                    )


                # TEXT
                elif file.type == "text/plain":

                    text = file_bytes.decode(
                        "utf-8",
                        errors="ignore"
                    )

                    inputs.append(
                        {
                            "type": "text",
                            "text": text
                        }
                    )


            # Add instructions last
            inputs.append(
                {
                    "type": "text",
                    "text": prompt
                }
            )


            # ------------------------------------------------
            # GEMINI INTERACTIONS API
            # ------------------------------------------------

            with st.spinner(
                "🤖 Gemini is analyzing the document..."
            ):

                interaction = client.interactions.create(
                    model=MODEL,
                    input=inputs,

                    response_format={
                        "type": "text",
                        "mime_type": "application/json",

                        "schema": {
                            "type": "object",

                            "properties": {

                                "patient_summary": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string"
                                        },
                                        "age": {
                                            "type": "string"
                                        },
                                        "gender": {
                                            "type": "string"
                                        },
                                        "summary": {
                                            "type": "string"
                                        }
                                    },
                                    "required": [
                                        "name",
                                        "age",
                                        "gender",
                                        "summary"
                                    ]
                                },

                                "diagnoses": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },

                                "medications": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },

                                "allergies": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },

                                "laboratory_results": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },

                                "clinical_findings": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },

                                "risk_flags": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "severity": {
                                                "type": "string"
                                            },
                                            "issue": {
                                                "type": "string"
                                            },
                                            "reason": {
                                                "type": "string"
                                            }
                                        },
                                        "required": [
                                            "severity",
                                            "issue",
                                            "reason"
                                        ]
                                    }
                                },

                                "recommended_next_step": {
                                    "type": "string"
                                },

                                "confidence": {
                                    "type": "string"
                                }
                            },

                            "required": [
                                "patient_summary",
                                "diagnoses",
                                "medications",
                                "allergies",
                                "laboratory_results",
                                "clinical_findings",
                                "risk_flags",
                                "recommended_next_step",
                                "confidence"
                            ]
                        }
                    }
                )


            # ------------------------------------------------
            # GET GEMINI RESPONSE
            # ------------------------------------------------

            result_text = interaction.output_text

            result = json.loads(result_text)


            # Save result
            st.session_state["result"] = result

            st.success(
                "✅ Clinical document analysis completed."
            )


        except Exception as e:

            st.error(
                "❌ Error while analyzing document"
            )

            st.code(str(e))


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.header("📊 Clinical Intelligence")


    # ========================================================
    # FOUR TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👤 Patient Summary",
            "🧪 Clinical Findings",
            "⚠️ Risk Flags",
            "📋 Recommendation"
        ]
    )


    # ========================================================
    # TAB 1 — PATIENT SUMMARY
    # ========================================================

    with tab1:

        patient = result["patient_summary"]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Patient",
                patient["name"]
            )

        with col2:

            st.metric(
                "Age",
                patient["age"]
            )

        with col3:

            st.metric(
                "Gender",
                patient["gender"]
            )


        st.subheader("Patient Summary")

        st.write(
            patient["summary"]
        )


        st.subheader("Diagnoses")

        if result["diagnoses"]:

            for diagnosis in result["diagnoses"]:

                st.write(
                    f"• {diagnosis}"
                )

        else:

            st.write("No diagnoses identified.")


        st.subheader("Medications")

        if result["medications"]:

            for medication in result["medications"]:

                st.write(
                    f"• {medication}"
                )

        else:

            st.write("No medications identified.")


        st.subheader("Allergies")

        if result["allergies"]:

            for allergy in result["allergies"]:

                st.write(
                    f"• {allergy}"
                )

        else:

            st.write("No allergies identified.")


    # ========================================================
    # TAB 2 — CLINICAL FINDINGS
    # ========================================================

    with tab2:

        st.subheader("🧪 Laboratory Results")

        if result["laboratory_results"]:

            for lab in result["laboratory_results"]:

                st.write(
                    f"• {lab}"
                )

        else:

            st.write(
                "No laboratory results identified."
            )


        st.subheader("🔎 Clinical Findings")

        if result["clinical_findings"]:

            for finding in result["clinical_findings"]:

                st.write(
                    f"• {finding}"
                )

        else:

            st.write(
                "No significant clinical findings identified."
            )


    # ========================================================
    # TAB 3 — RISK FLAGS
    # ========================================================

    with tab3:

        risks = result["risk_flags"]

        if risks:

            for risk in risks:

                severity = risk["severity"]

                issue = risk["issue"]

                reason = risk["reason"]


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
                        f"🟡 {severity.upper()} — {issue}\n\n"
                        f"{reason}"
                    )

        else:

            st.success(
                "✅ No potential risk flags identified."
            )


    # ========================================================
    # TAB 4 — RECOMMENDATION
    # ========================================================

    with tab4:

        st.subheader(
            "📋 Recommended Next Step"
        )

        st.info(
            result["recommended_next_step"]
        )


        st.subheader(
            "AI Confidence"
        )

        st.write(
            result["confidence"]
        )


    # ========================================================
    # RAW JSON
    # ========================================================

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
    "information extraction and decision support. "
    "It does not replace professional medical judgment. "
    "Use synthetic or publicly available clinical data."
)