
# # # import streamlit as st
# # # import json

# # # st.set_page_config(page_title="RLHF Task Inspector", layout="wide")
# # # st.title("🧠 RLHF Task Viewer")

# # # # --- Upload File ---
# # # st.sidebar.header("📁 Upload RLHF JSON File")
# # # json_file = st.sidebar.file_uploader("Upload delivery JSON", type="json")

# # # if not json_file:
# # #     st.info("👈 Upload a JSON file to begin.")
# # #     st.stop()

# # # # --- Load JSON ---
# # # try:
# # #     data = json.load(json_file)
# # # except Exception as e:
# # #     st.error(f"❌ Failed to load JSON: {e}")
# # #     st.stop()

# # # if "rlhf" not in data:
# # #     st.error("❌ Uploaded JSON does not contain a top-level `rlhf` key.")
# # #     st.stop()

# # # rlhf_tasks = data["rlhf"]
# # # task_id_map = {str(task["taskId"]): task for task in rlhf_tasks if "taskId" in task}

# # # if not task_id_map:
# # #     st.warning("⚠️ No valid RLHF tasks with `taskId` found.")
# # #     st.stop()

# # # # --- Task Dropdown ---
# # # selected_task_id = st.selectbox("🔍 Select a Task ID to View", list(task_id_map.keys()))

# # # if selected_task_id:
# # #     task = task_id_map[selected_task_id]
# # #     st.subheader(f"📄 Task ID: {selected_task_id}")

# # #     colab_link = task.get("task", {}).get("colabLink", "N/A")
# # #     st.markdown(f"🔗 **RLHF Link**: [Open Task]({colab_link})")

# # #     # --- Scope Requirements ---
# # #     st.markdown("### 📘 Scope Requirements")
# # #     scope = task.get("metadata", {}).get("scope_requirements", {})
# # #     if scope:
# # #         scope_table = [{"Key": k, "Value": v} for k, v in scope.items()]
# # #         st.table(scope_table)
# # #     else:
# # #         st.info("No scope_requirements found.")

# # #     # --- User Prompt & Message Details ---
# # #     st.markdown("### 💬 Prompt & Other User Message Fields")

# # #     messages = task.get("messages", [])
# # #     user_msg = next((m for m in messages if m.get("role") == "user"), None)

# # #     if user_msg:
# # #         st.markdown(f"**Prompt (text)**: {user_msg.get('text', '')}")
# # #         st.markdown("---")

# # #         other_keys = {k: v for k, v in user_msg.items() if k != "text"}
# # #         # for key, val in other_keys.items():
# # #         #     st.markdown(f"#### 🔹 {key}")
# # #         #     st.json(val)
# # #         for key, val in other_keys.items():
# # #             st.markdown(f"#### 🔹 {key}")
# # #             try:
# # #                 st.json(val)
# # #             except Exception:
# # #                 st.write(val)

# # #     else:
# # #         st.warning("⚠️ No user message found in messages.")
# # import streamlit as st
# # import json
# # import pandas as pd

# # st.set_page_config(page_title="RLHF Task Inspector", layout="wide")
# # st.title("🧠 RLHF Task Viewer")

# # # --- Upload File ---
# # st.sidebar.header("📁 Upload RLHF JSON File")
# # json_file = st.sidebar.file_uploader("Upload delivery JSON", type="json")

# # if not json_file:
# #     st.info("👈 Upload a JSON file to begin.")
# #     st.stop()

# # # --- Load JSON ---
# # try:
# #     data = json.load(json_file)
# # except Exception as e:
# #     st.error(f"❌ Failed to load JSON: {e}")
# #     st.stop()

# # if "rlhf" not in data:
# #     st.error("❌ Uploaded JSON does not contain a top-level `rlhf` key.")
# #     st.stop()

# # rlhf_tasks = data["rlhf"]
# # task_id_map = {str(task["taskId"]): task for task in rlhf_tasks if "taskId" in task}

# # if not task_id_map:
# #     st.warning("⚠️ No valid RLHF tasks with `taskId` found.")
# #     st.stop()

# # # --- Task Dropdown ---
# # selected_task_id = st.selectbox("🔍 Select a Task ID to View", list(task_id_map.keys()))

# # if selected_task_id:
# #     task = task_id_map[selected_task_id]
# #     st.subheader(f"📄 Task ID: {selected_task_id}")

# #     # --- RLHF Link ---
# #     colab_link = task.get("task", {}).get("colabLink", "N/A")
# #     st.markdown(f"🔗 **RLHF Link**: [Open Task]({colab_link})")

# #     # --- Scope Requirements ---
# #     st.markdown("### 📘 Scope Requirements")
# #     scope = task.get("metadata", {}).get("scope_requirements", {})
# #     if scope:
# #         scope_table = [{"Key": k, "Value": str(v)} for k, v in scope.items()]  # convert all values to string
# #         st.table(pd.DataFrame(scope_table))
# #     else:
# #         st.info("No scope_requirements found.")

# #     # --- User Prompt & Message Details ---
# #     st.markdown("### 💬 Prompt & Other User Message Fields")

# #     messages = task.get("messages", [])
# #     user_msg = next((m for m in messages if m.get("role") == "user"), None)

# #     if user_msg:
# #         prompt_text = user_msg.get("text", "")
# #         st.markdown(f"**Prompt (text):**")
# #         st.code(prompt_text)

# #         st.markdown("---")
# #         st.markdown("### 📦 Other Fields in User Message")

# #         other_keys = {k: v for k, v in user_msg.items() if k != "text"}

# #         for key, val in other_keys.items():
# #             st.markdown(f"#### 🔹 {key}")

# #             if key == "prompt_evaluation" and isinstance(val, list):
# #                 # Format as table
# #                 st.markdown("**Prompt Evaluation Summary:**")
# #                 table_rows = []
# #                 for entry in val:
# #                     q = entry.get("question", "")
# #                     d = entry.get("description", "")
# #                     v = entry.get("human_input_value", "")
# #                     table_rows.append({"Question": q, "Description": d, "Human Answer": v})

# #                 if table_rows:
# #                     st.table(pd.DataFrame(table_rows))
# #                 else:
# #                     st.info("No prompt evaluation entries found.")
# #             else:
# #                 try:
# #                     if isinstance(val, (dict, list)):
# #                         st.json(val)
# #                     else:
# #                         st.write(val)
# #                 except Exception:
# #                     st.write(str(val))


# #     else:
# #         st.warning("⚠️ No user message found in messages.")


# import streamlit as st
# import json
# import pandas as pd

# st.set_page_config(page_title="RLHF Task Inspector", layout="wide")
# st.title("🧠 RLHF Task Viewer")

# # --- Upload File ---
# st.sidebar.header("📁 Upload RLHF JSON File")
# json_file = st.sidebar.file_uploader("Upload delivery JSON", type="json")

# if not json_file:
#     st.info("👈 Upload a JSON file to begin.")
#     st.stop()

# # --- Load JSON ---
# try:
#     data = json.load(json_file)
# except Exception as e:
#     st.error(f"❌ Failed to load JSON: {e}")
#     st.stop()

# if "rlhf" not in data:
#     st.error("❌ Uploaded JSON does not contain a top-level `rlhf` key.")
#     st.stop()

# rlhf_tasks = data["rlhf"]
# task_id_map = {str(task["taskId"]): task for task in rlhf_tasks if "taskId" in task}

# if not task_id_map:
#     st.warning("⚠️ No valid RLHF tasks with `taskId` found.")
#     st.stop()

# # --- Task Dropdown ---
# selected_task_id = st.selectbox("🔍 Select a Task ID to View", list(task_id_map.keys()))

# if selected_task_id:
#     task = task_id_map[selected_task_id]
#     st.subheader(f"📄 Task ID: {selected_task_id}")

#     # --- RLHF Link ---
#     colab_link = task.get("task", {}).get("colabLink", "N/A")
#     st.markdown(f"🔗 **RLHF Link**: [Open Task]({colab_link})")

#     # --- Scope Requirements ---
#     st.markdown("### 📘 Scope Requirements")
#     scope = task.get("metadata", {}).get("scope_requirements", {})
#     if scope:
#         scope_table = [{"Key": k, "Value": str(v)} for k, v in scope.items()]
#         st.table(pd.DataFrame(scope_table))
#     else:
#         st.info("No scope_requirements found.")

#     # --- Message Rendering ---
#     st.markdown("## 💬 Messages")

#     messages = task.get("messages", [])

#     for msg in messages:
#         role = msg.get("role", "").capitalize()
#         st.markdown(f"### 🔹 Role: `{role}`")

#         # --- User Role ---
#         if role == "User":
#             prompt_text = msg.get("text", "")
#             st.markdown("**Prompt (text):**")
#             st.code(prompt_text)

#             st.markdown("### 📦 Other Fields in User Message")
#             other_keys = {k: v for k, v in msg.items() if k not in ["text", "role"]}

#             for key, val in other_keys.items():
#                 st.markdown(f"#### 🔹 {key}")
#                 if key == "prompt_evaluation" and isinstance(val, list):
#                     table_rows = [
#                         {
#                             "Question": entry.get("question", ""),
#                             "Description": entry.get("description", ""),
#                             "Human Answer": entry.get("human_input_value", "")
#                         }
#                         for entry in val
#                     ]
#                     st.table(pd.DataFrame(table_rows))
#                 else:
#                     try:
#                         if isinstance(val, (dict, list)):
#                             st.json(val)
#                         else:
#                             st.write(val)
#                     except Exception:
#                         st.write(str(val))

#         # --- Assistant Role ---
#         elif role == "Assistant":
#             signal = msg.get("signal", {})

#             # Ideal Response
#             ideal_response = signal.get("ideal_response")
#             if ideal_response:
#                 st.markdown("**💡 Ideal Response:**")
#                 st.code(ideal_response)

#             # Human Evaluations
#             human_evals = signal.get("human_evals", [])
#             for eval_set in human_evals:
#                 eval_form = eval_set.get("evaluation_form", [])
#                 if eval_form:
#                     st.markdown("**🧾 Human Evaluation Table:**")
#                     rows = [
#                         {
#                             "Question": item.get("question", ""),
#                             "Description": item.get("description", ""),
#                             "Human Answer": item.get("human_input_value", "")
#                         }
#                         for item in eval_form
#                     ]
#                     st.table(pd.DataFrame(rows))
import streamlit as st
import json
import pandas as pd
import io

def load_json(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return json.load(uploaded_file)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return None

st.set_page_config(page_title="RLHF Task Inspector", layout="wide")
st.title("🧠 RLHF Task Viewer")

# --- Upload File ---
st.sidebar.header("📁 Upload RLHF JSON File")
# Main RLHF delivery JSON
json_file = st.sidebar.file_uploader(
    "Upload RLHF JSON",
    type="json",
    key="delivery_json_uploader",
    help="Primary RLHF delivery payload."
)

# Additional ingestion file
ingestion_file = st.sidebar.file_uploader(
    "Upload ingestion JSON",
    type="json",   # change to ["json","csv","xlsx"] if needed
    key="ingestion_json_uploader",
    help="Supplemental ingestion metadata or records."
)


if not json_file:
    st.info("👈 Upload a JSON file to begin.")
    st.stop()

# --- Load JSON ---
try:
    data = json.load(json_file)
except Exception as e:
    st.error(f"❌ Failed to load JSON: {e}")
    st.stop()

ingestion_data = load_json(ingestion_file)

st.subheader("File Load Status")
col1, col2 = st.columns(2)
with col1:
    st.write("**Delivery JSON:**", "✅ Loaded" if data is not None else "— Not loaded —")
with col2:
    st.write("**Ingestion JSON:**", "✅ Loaded" if ingestion_data is not None else "— Not loaded —")


if "rlhf" not in data:
    st.error("❌ Uploaded JSON does not contain a top-level `rlhf` key.")
    st.stop()

rlhf_tasks = data["rlhf"]
data_sft = data["sft"]
annotator_task_dict = {}
for task_ in data_sft:
    task_id = task_.get("id")
    annotator_id = task_.get("humanRoleId")
    if annotator_id in annotator_task_dict.keys():
        annotator_task_dict[annotator_id].append(task_id)
    else:
        annotator_task_dict[annotator_id] = [task_id]

annotator_task_df = pd.Series(annotator_task_dict).explode().reset_index(name='task_id').rename(columns={'index': 'AnnotatorID'})
# Ensure numeric dtype if appropriate
annotator_task_df['task_id'] = pd.to_numeric(annotator_task_df['task_id'], errors='ignore')

task_id_map = {str(task["taskId"]): task for task in rlhf_tasks if "taskId" in task}

if not task_id_map:
    st.warning("⚠️ No valid RLHF tasks with `taskId` found.")
    st.stop()

# --- Task Dropdown ---
selected_task_id = st.selectbox("🔍 Select a Task ID to View", list(task_id_map.keys()))

# --- Task-Level UI Display ---
if selected_task_id:
    task = task_id_map[selected_task_id]
    st.subheader(f"📄 Task ID: {selected_task_id}")

    # --- RLHF Link ---
    colab_link = task.get("task", {}).get("colabLink", "N/A")
    st.markdown(f"🔗 **RLHF Link**: [Open Task]({colab_link})")

    # --- Scope Requirements ---
    st.markdown("### 📘 Scope Requirements")
    scope = task.get("metadata", {}).get("scope_requirements", {})
    if scope:
        scope_table = [{"Key": k, "Value": str(v)} for k, v in scope.items()]
        print(f"Data: {selected_task_id}")
        print(type(selected_task_id))
        annotator_id_ = annotator_task_df[annotator_task_df["task_id"]==int(selected_task_id)]["AnnotatorID"].tolist()
        if annotator_id_:
            id_ = annotator_id_[0]
        else:
            id_ = "NA"
        scope_table.append({"Key": "annotator_id", "Value":id_})
        st.table(pd.DataFrame(scope_table))
    else:
        st.info("No scope_requirements found.")

    # --- Messages Display ---
    st.markdown("## 💬 Messages")
    messages = task.get("messages", [])

    for msg in messages:
        role = msg.get("role", "").capitalize()
        st.markdown(f"### 🔹 Role: `{role}`")

        if role == "User":
            prompt_text = msg.get("text", "")
            st.markdown("**Prompt (text):**")
            st.code(prompt_text)

            st.markdown("### 📦 Other Fields in User Message")
            for key, val in msg.items():
                if key in ["text", "role"]:
                    continue

                st.markdown(f"#### 🔹 {key}")
                if key == "prompt_evaluation" and isinstance(val, list):
                    rows = [
                        {
                            "Question": item.get("question", ""),
                            "Description": item.get("description", ""),
                            "Human Answer": item.get("human_input_value", "")
                        } for item in val
                    ]
                    st.table(pd.DataFrame(rows))
                else:
                    try:
                        if isinstance(val, (dict, list)):
                            st.json(val)
                        else:
                            st.write(val)
                    except:
                        st.write(str(val))

        elif role == "Assistant":
            signal = msg.get("signal", {})

            # Ideal Response
            ideal_response = signal.get("ideal_response")
            if ideal_response:
                st.markdown("**💡 Ideal Response:**")
                st.code(ideal_response)

            # Human Evals
            human_evals = signal.get("human_evals", [])
            for eval_set in human_evals:
                eval_form = eval_set.get("evaluation_form", [])
                if eval_form:
                    st.markdown("**🧾 Human Evaluation Table:**")
                    rows = [
                        {
                            "Question": item.get("question", ""),
                            "Description": item.get("description", ""),
                            "Human Answer": item.get("human_input_value", "")
                        } for item in eval_form
                    ]
                    st.table(pd.DataFrame(rows))

# -----------------------------------------
# ✅ Compile All Tasks and Offer CSV Download
# -----------------------------------------
st.markdown("## 📥 Export All Tasks as CSV")

compiled_rows = []

for task_id, task in task_id_map.items():
    row = {
        "task_id": task_id,
        "colab_link": task.get("task", {}).get("colabLink", "")
    }

    # Scope fields
    scope = task.get("metadata", {}).get("scope_requirements", {})
    for k, v in scope.items():
        row[f"scope_{k}"] = v

    # Messages
    user_msg = next((m for m in task.get("messages", []) if m.get("role") == "user"), None)
    assistant_msg = next((m for m in task.get("messages", []) if m.get("role") == "assistant"), None)

    # User Prompt & Evaluation
    if user_msg:
        row["prompt"] = user_msg.get("text", "")

        for entry in user_msg.get("prompt_evaluation", []):
            q = entry.get("question", "").strip().replace(" ", "_").lower()
            row[f"prompt_eval_{q}"] = entry.get("human_input_value", "")

    # Assistant Ideal Response & Evaluation
    if assistant_msg:
        signal = assistant_msg.get("signal", {})
        row["ideal_response"] = signal.get("ideal_response", "")

        for eval_set in signal.get("human_evals", []):
            for item in eval_set.get("evaluation_form", []):
                q = item.get("question", "").strip().replace(" ", "_").lower()
                row[f"assistant_eval_{q}"] = item.get("human_input_value", "")

    compiled_rows.append(row)

if compiled_rows:
    invalid_task_ids = []
    df = pd.DataFrame(compiled_rows)
    df["task_id"] = pd.to_numeric(df["task_id"], errors="raise")
    df = df.merge(annotator_task_df, on="task_id", how="left")
    st.dataframe(df, use_container_width=True)
    data_csv = df.copy()
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    ingestion_loaded = ingestion_data is not None

    if ingestion_loaded:
        import uuid
        from datetime import datetime, timezone
        import pandas as pd
        import numpy as np

        # --- time util ----------------------------------------------------------------
        def current_utc_iso_millis() -> str:
            now = datetime.now(timezone.utc)
            return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # --- metadata cols (all columns that contain 'eval' anywhere, case-insensitive) -
        eval_cols = [col for col in data_csv.columns if "eval" in col.lower()]

        # --- build one annotation record from a row -----------------------------------
        def row_to_record(row, eval_cols):
            # metadata subset
            meta = {col: row[col] for col in eval_cols if col in row.index}
            # NaN -> None for JSON
            meta = {k: (None if pd.isna(v) else v) for k, v in meta.items()}

            question = row["prompt"] if "prompt" in row and not pd.isna(row["prompt"]) else None
            answer   = row["ideal_response"] if "ideal_response" in row and not pd.isna(row["ideal_response"]) else None

            return {
                "question": question,
                "answer": answer,
                "metadata": meta,
            }

        # --- main build ---------------------------------------------------------------
        sft_annotation = []
        for task in data_sft:
            subset = {k: task.get(k) for k in ("createdAt", "id", "humanRoleId")}
            sft_annotation.append(subset)

        sft_annotation_df = pd.DataFrame(sft_annotation)


        master_list = []
        work_item_ids = [workItemId.get("workItemId") for workItemId in ingestion_data.get("workitems")]
        deliveries_gen = min(data_csv["scope_pdf_link"].nunique(), len(work_item_ids))
        work_item_ids_needed = work_item_ids[:deliveries_gen]

        pdf_unique = data_csv["scope_pdf_link"].dropna().unique().tolist()

        for index, work_id in enumerate(work_item_ids_needed):
            if index >= len(pdf_unique):
                break  # safety: fewer PDFs than work items

            pdf_name = pdf_unique[index]
            data_tmp = data_csv.loc[data_csv["scope_pdf_link"] == pdf_name].reset_index(drop=True)
            if data_tmp.shape[0]==3:
                # annotator (first non-null)
                annot_vals = data_tmp["AnnotatorID"].dropna().unique().tolist() if "AnnotatorID" in data_tmp else []
                annotator_id = str(annot_vals[0]) if annot_vals else "NA"

                # annotations for this PDF
                records = [row_to_record(r, eval_cols) for _, r in data_tmp.iterrows()]
                task_answers = {"id": pdf_name, "annotations": records}
                task_answers_dict = {"taskAnswers": [task_answers]}
                variation_task_ids = data_tmp["task_id"].unique().tolist()
                current_utc_iso_millis = sft_annotation_df[sft_annotation_df["id"].isin(variation_task_ids)]["createdAt"].tolist()[-1]
                # metadata schema (per task)
                metadata_schema = {
                    "taskId": str(uuid.uuid4()),              # unique per PDF/task
                    "operationType": "LABELLING",
                    "labelledTimestamp": current_utc_iso_millis,
                    "obfuscatedDaAlias": annotator_id,
                }

                uid_data = {"data": task_answers_dict, "metadata": metadata_schema}

                # outer wrapper (structure preserved from your code)
                tmp_dict = {
                    "workItemId": work_id,
                    "workflow": "workflow_name",
                    "locale": "en_US",
                    "inputData": {"Document": pdf_name},
                    "metadata": {},
                    str(uuid.uuid4()): [uid_data],            # random key bucket w/ list payload
                }

                master_list.append(tmp_dict)
            

        # remove this break when ready to process all items


        final_json_ls = {"fileMetadata": ingestion_data.get("fileMetadata"), "workitems":master_list}
        final_json_str = json.dumps(final_json_ls, ensure_ascii=False, indent=2)
        final_json_bytes = final_json_str.encode("utf-8")
    else:
        final_json_bytes = b"" 

    def timestamped_json_filename(prefix="rlhf_final_payload", suffix=".json"):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}_{ts}{suffix}"

    fname = timestamped_json_filename()

    st.download_button(
        label="📄 Download All Task Data as CSV",
        data=csv_buf.getvalue(),
        file_name="rlhf_tasks_export.csv",
        mime="text/csv"
    )

    st.download_button(
    label="📦 Download Final JSON",
    data=final_json_bytes,
    file_name=fname,
    mime="application/json",
    disabled=not ingestion_loaded,    # <-- key line
    help="Upload an ingestion file to enable this download.",
    key="download_final_json_btn"
)