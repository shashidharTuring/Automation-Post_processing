
import streamlit as st
import json
import os
from dotenv import load_dotenv
import openai

# Load OpenAI API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found in .env file")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="JSON Visualizer & Explainer", layout="wide")
st.title("📊 JSON Visualizer & Explainer (Powered by GPT-4o)")

# ---- Sidebar: Upload a single JSON file ----
st.sidebar.header("📁 Upload Your JSON File")
json_file = st.sidebar.file_uploader("Upload JSON File", type="json")

# ---- Load and display the JSON ----
json_data = None
if json_file:
    try:
        json_data = json.load(json_file)
    except Exception as e:
        st.error(f"❌ Failed to read JSON: {str(e)}")

if json_data:
    st.subheader("📄 Uploaded JSON")
    st.json(json_data)

    with st.spinner("Analyzing uploaded JSON in context with GPT-4o..."):
        try:
            sample_summary = json.dumps(json_data, indent=2)
            if len(sample_summary) > 10000:
                sample_summary = sample_summary[:10000] + "\n...\n[TRUNCATED FOR ANALYSIS]"

            prompt = f"""
You are a JSON specialist. Please deeply analyze the *uploaded JSON* (provided below). Your output must follow this format:

### 1️⃣ JSON Format & Structure
- Describe the structure: whether it's an object, array, or list of objects
- Mention relationships between nested fields

### 2️⃣ Key Details Table
Prepare a table with the following columns:
| Object | Key | Data Type | Required (Y/N) |
|--------|-----|-----------|----------------|
(Fill in with keys from uploaded JSON and inferred data types. Mark keys used across all records as Required.)

### 3️⃣ Purpose of Each Field
- List and explain each key (e.g., `task_id`: uniquely identifies the task, `metadata.pdf_name`: name of the source file, etc.)

### 4️⃣ Data Quality Assessment
- Mention any observed issues like incorrect types, inconsistencies, overly verbose text, missing fields, etc.

### 5️⃣ Cleaned JSON
- Provide a cleaner and best-practice version of the uploaded JSON using only the visible portion provided below.

```json
{sample_summary}
```
"""

            # response = client.chat.completions.create(
            #     model="gpt-4o",
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.7,
            #     max_tokens=4096,
            #     stream=True
            # )
            response = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system", "content": (
                        "You are a reasoning expert and a JSON analysis specialist. "
                        "Your job is to deeply analyze uploaded JSON data with step-by-step logic, clear formatting, and structured breakdowns."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
                stream=True
            )


            full_output = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_output += chunk.choices[0].delta.content

            st.markdown("### 🧠 Uploaded JSON: Full Breakdown and Cleaned Version")
            st.markdown(full_output)

        except Exception as e:
            st.error(f"❌ Failed to analyze JSON due to: {str(e)}")
else:
    st.info("📥 Upload a JSON file using the sidebar to begin analysis.")
