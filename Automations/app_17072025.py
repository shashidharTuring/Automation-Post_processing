import streamlit as st
import json
import pandas as pd
import numpy as np
import io
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv
import openai

# =============================================================================
# CUSTOM CSS (responsive width + visual polish)
# =============================================================================
st.set_page_config(page_title="RLHF & JSON Viewer", layout="wide")
CUSTOM_CSS = """
<style>
:root {
  --rlhf-blue: #1E88E5;
  --rlhf-green: #43A047;
  --rlhf-purple: #8E24AA;
  --rlhf-amber: #FDD835;
  --rlhf-gray-bg: #f7f9fc;
  --rlhf-gray-border: #dfe2e8;
  --rlhf-card-radius: 8px;
  --rlhf-card-pad: 0.75rem 1rem;
  --rlhf-font-sm: 0.86rem;
  --rlhf-font-xs: 0.75rem;
}

/* widen main content */
main .block-container {
  max-width: 95vw !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}

/* sidebar */
[data-testid="stSidebar"] {
  background: var(--rlhf-gray-bg);
  border-right: 1px solid var(--rlhf-gray-border);
}

/* chips */
.rlhf-chip {
  display: inline-block;
  padding: 2px 8px;
  margin: 0 4px 4px 0;
  font-size: var(--rlhf-font-xs);
  border-radius: 12px;
  background: var(--rlhf-blue);
  color: white;
  white-space: nowrap;
}
.rlhf-chip-green { background: var(--rlhf-green); }
.rlhf-chip-purple { background: var(--rlhf-purple); }
.rlhf-chip-amber { background: var(--rlhf-amber); color: #333; }

/* kpi bar */
.rlhf-kpi-value {
  font-weight: 600;
  font-size: 1.05rem;
  margin-bottom: 0;
  line-height: 1.2;
}
.rlhf-kpi-label {
  font-size: var(--rlhf-font-xs);
  color: #666;
  margin-top: -2px;
}

/* role badges */
.rlhf-badge-user, .rlhf-badge-assistant {
  padding: 0 6px;
  border-radius: 6px;
  font-size: var(--rlhf-font-xs);
}
.rlhf-badge-user { background: var(--rlhf-blue); color: #fff; }
.rlhf-badge-assistant { background: var(--rlhf-green); color: #fff; }

/* code blocks */
.rlhf-code-scroll pre, code {
  max-width: 100%;
  white-space: pre-wrap;
  word-break: break-word;
}
.rlhf-code-scroll pre {
  max-height: 260px;
  overflow-y: auto;
  overflow-x: auto;
}

/* sticky toolbar */
.rlhf-toolbar-sticky {
  position: sticky;
  top: 48px;
  z-index: 49;
  padding: 0.5rem;
  background: #ffffffeb;
  backdrop-filter: blur(4px);
  border-bottom: 1px solid var(--rlhf-gray-border);
  margin-bottom: 0.5rem;
}

/* delivery summary */
.rlhf-delivery-summary {
  font-size: var(--rlhf-font-sm);
  margin-bottom: 0.5rem;
}
.rlhf-delivery-summary strong { color: var(--rlhf-blue); }

/* validator dropzone */
.rlhf-dropzone {
  padding: 2rem;
  border: 2px dashed var(--rlhf-gray-border);
  border-radius: var(--rlhf-card-radius);
  background: var(--rlhf-gray-bg);
  text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# MAIN PAGE TABS
# =============================================================================
st.title("🔍 Choose Viewer Mode")
option = st.radio(
    label="Select a Viewer",
    options=["Home", "RLHF Viewer", "JSON Visualizer"],
    index=0,
    horizontal=True
)



# =============================================================================
# BASIC UTILITIES
# =============================================================================
def load_json(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return json.load(uploaded_file)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to parse JSON: {e}")
        return None


def current_utc_iso_millis() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def timestamped_json_filename(prefix="rlhf_final_payload", suffix=".json") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}{suffix}"


def build_annotator_task_df(data_sft: List[Dict[str, Any]]) -> pd.DataFrame:
    annotator_task_dict = {}
    for task_ in data_sft:
        task_id = task_.get("id")
        annotator_id = task_.get("humanRoleId")
        annotator_task_dict.setdefault(annotator_id, []).append(task_id)

    annotator_task_df = (
        pd.Series(annotator_task_dict)
        .explode()
        .reset_index(name="task_id")
        .rename(columns={"index": "AnnotatorID"})
    )
    annotator_task_df["task_id_orig"] = annotator_task_df["task_id"]
    annotator_task_df["task_id"] = pd.to_numeric(
        annotator_task_df["task_id"], errors="coerce"
    )
    return annotator_task_df


def compile_task_rows(
    task_id_map: Dict[str, Dict[str, Any]],
    data_sft: List[Dict[str, Any]],
    annotator_task_df: pd.DataFrame,
) -> pd.DataFrame:
    compiled_rows: List[Dict[str, Any]] = []

    for task_id, task in task_id_map.items():
        row: Dict[str, Any] = {
            "task_id": task_id,
            "colab_link": task.get("task", {}).get("colabLink", ""),
        }

        scope = task.get("metadata", {}).get("scope_requirements", {})
        for k, v in scope.items():
            row[f"scope_{k}"] = v

        user_msg = next(
            (m for m in task.get("messages", []) if m.get("role") == "user"), None
        )
        assistant_msg = next(
            (m for m in task.get("messages", []) if m.get("role") == "assistant"), None
        )

        if user_msg:
            row["prompt"] = user_msg.get("text", "")
            for entry in user_msg.get("prompt_evaluation", []):
                q = entry.get("question", "").strip().replace(" ", "_").lower()
                row[f"prompt_eval_{q}"] = entry.get("human_input_value", "")

        if assistant_msg:
            signal = assistant_msg.get("signal", {})
            row["ideal_response"] = signal.get("ideal_response", "")
            for eval_set in signal.get("human_evals", []):
                for item in eval_set.get("evaluation_form", []):
                    q = item.get("question", "").strip().replace(" ", "_").lower()
                    row[f"assistant_eval_{q}"] = item.get("human_input_value", "")

        compiled_rows.append(row)

    if not compiled_rows:
        return pd.DataFrame()

    df = pd.DataFrame(compiled_rows)
    df["task_id_orig"] = df["task_id"]
    df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")
    df = df.merge(annotator_task_df, on="task_id", how="left")
    return df


def _safe_show_eval_rows(rows: List[Dict[str, Any]]):
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.applymap(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        df = df.astype(str)
    st.table(df)


# =============================================================================
# FILTERS (multi-select, targeted cols)
# =============================================================================
ALLOWED_FILTER_COLS = ["task_id", "scope_category", "scope_batchName", "scope_batchId"]


def filter_dataframe(
    df: pd.DataFrame, allowed_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    if allowed_cols is None:
        allowed_cols = ALLOWED_FILTER_COLS

    out = df.copy()
    cols_present = [c for c in allowed_cols if c in df.columns]
    cols_missing = [c for c in allowed_cols if c not in df.columns]
    if cols_missing:
        st.caption("Missing filter columns: " + ", ".join(cols_missing))
    if not cols_present:
        return out

    # 2-column grid for widgets
    n_cols = 2
    col_widgets = st.columns(n_cols)
    sel_summary = []

    for i, col in enumerate(cols_present):
        idx = i % n_cols
        container = col_widgets[idx]
        with container:
            series = df[col]
            non_null = series.dropna()

            uniq_vals = non_null.unique().tolist()
            disp_map = {str(v): v for v in uniq_vals}
            disp_labels = sorted(disp_map.keys(), key=lambda x: (len(x), x))

            sel_labels = st.multiselect(
                col,
                options=disp_labels,
                default=[],
                key=f"flt_ms_{col}",
                help=f"Filter rows where {col} matches selected values.",
            )
            if sel_labels:
                sel_vals = [disp_map[l] for l in sel_labels]
                out = out[out[col].isin(sel_vals)]
                sel_summary.append((col, sel_labels))

    if sel_summary:
        chip_html = ""
        for col, labels in sel_summary:
            for val in labels:
                chip_html += f"<span class='rlhf-chip-amber'>{col}: {val}</span>"
        st.markdown("**Active filters:** " + chip_html, unsafe_allow_html=True)

    return out


# =============================================================================
# DELIVERY HELPERS
# =============================================================================
def row_to_record(row: pd.Series, eval_cols: List[str]) -> Dict[str, Any]:
    print("Inside row_to_record")
    meta = {col: row[col] for col in eval_cols if col in row.index}
    meta = {k: (None if pd.isna(v) else v) for k, v in meta.items()}

    question = (
        row["prompt"] if "prompt" in row and not pd.isna(row["prompt"]) else None
    )
    print(f"Question: {question}")
    answer = (
        row["ideal_response"]
        if "ideal_response" in row and not pd.isna(row["ideal_response"])
        else None
    )
    return {"question": question, "answer": answer, "metadata": meta}


def build_delivery_json_from_ingestion(
    ingestion_data: Dict[str, Any],
    data_csv: pd.DataFrame,
    data_sft: List[Dict[str, Any]],
    pdf_col: str = "scope_pdf_name",
) -> Dict[str, Any]:
    if ingestion_data is None:
        return {}

    sft_annotation = []
    for task in data_sft:
        subset = {k: task.get(k) for k in ("createdAt", "id", "humanRoleId")}
        sft_annotation.append(subset)
    sft_annotation_df = pd.DataFrame(sft_annotation)

    if pdf_col not in data_csv.columns:
        st.warning(
            f"Column `{pdf_col}` not found; using single dummy group for delivery batch build."
        )
        data_csv = data_csv.copy()
        data_csv[pdf_col] = "_NO_PDF_KEY_"

    work_item_ids = [w.get("workItemId") for w in ingestion_data.get("workitems", [])]
    if not work_item_ids:
        st.error("Ingestion JSON has no `workitems` with `workItemId`.")
        return {}

    pdf_unique = data_csv[pdf_col].dropna().unique().tolist()
    deliveries_gen = min(len(pdf_unique), len(work_item_ids))
    work_item_ids_needed = work_item_ids[:deliveries_gen]

    eval_cols = [col for col in data_csv.columns if "eval" in col.lower()]

    master_list = []
    skipped_groups = []
    Counter_1 = 0
    counter_2 = 0
    for index, work_id in enumerate(work_item_ids_needed):
        if index >= len(pdf_unique):
            break
        pdf_name = pdf_unique[index]
        data_tmp = data_csv.loc[data_csv[pdf_col] == pdf_name].reset_index(drop=True)
        Counter_1+=1
        if data_tmp.shape[0] < 3:
            counter_2+=1
            skipped_groups.append((pdf_name, data_tmp.shape[0]))
            continue

        annot_vals = (
            data_tmp["AnnotatorID"].dropna().unique().tolist()
            if "AnnotatorID" in data_tmp
            else []
        )
        annotator_id = str(annot_vals[0]) if annot_vals else "NA"

        records = [row_to_record(r, eval_cols) for _, r in data_tmp.iterrows()]
        task_answers = {"id": pdf_name, "annotations": records}
        task_answers_dict = {"taskAnswers": [task_answers]}

        variation_task_ids = data_tmp["task_id"].unique().tolist()
        try:
            labelled_ts = (
                sft_annotation_df[sft_annotation_df["id"].isin(variation_task_ids)][
                    "createdAt"
                ]
                .tolist()[-1]
            )
        except Exception:  # noqa: BLE001
            labelled_ts = current_utc_iso_millis()

        metadata_schema = {
            "taskId": str(uuid.uuid4()),
            "operationType": "LABELLING",
            "labelledTimestamp": labelled_ts,
            "obfuscatedDaAlias": annotator_id,
        }

        uid_data = {"data": task_answers_dict, "metadata": metadata_schema}
        tmp_dict = {
            "workItemId": work_id,
            "workflow": "workflow_name",
            "locale": "en_US",
            "inputData": {"Document": pdf_name},
            "metadata": {},
            str(uuid.uuid4()): [uid_data],
        }
        master_list.append(tmp_dict)

    if skipped_groups:
        # st.warning(
        #     "Skipped groups (rows != 3): "
        #     + ", ".join([f"{name}({n})" for name, n in skipped_groups])
        # )
        pass

    return {
        "fileMetadata": ingestion_data.get("fileMetadata"),
        "workitems": master_list,
    }


# =============================================================================
# TABS
# =============================================================================
if option == "RLHF Viewer":
    st.title("🧠 RLHF Task Viewer")

    # =============================================================================
    # SIDEBAR – DELIVERY RLHF JSON
    # =============================================================================
    st.sidebar.header("📁 Upload RLHF Delivery JSON")
    json_file = st.sidebar.file_uploader(
        "Upload delivery JSON", type="json", key="delivery_json_uploader"
    )

    if not json_file:
        st.info("👈 Upload a Delivery RLHF JSON to begin.")
        st.stop()

    data = load_json(json_file)
    if data is None:
        st.stop()

    if "rlhf" not in data:
        st.error("❌ Uploaded JSON missing top-level `rlhf` key.")
        st.stop()
    if "sft" not in data:
        st.error("❌ Uploaded JSON missing top-level `sft` key.")
        st.stop()

    rlhf_tasks = data["rlhf"]
    data_sft = data["sft"]

    annotator_task_df = build_annotator_task_df(data_sft)
    task_id_map = {str(task["taskId"]): task for task in rlhf_tasks if "taskId" in task}
    if not task_id_map:
        st.warning("⚠️ No valid RLHF tasks with `taskId` found.")
        st.stop()

    data_csv = compile_task_rows(task_id_map, data_sft, annotator_task_df)


    # =============================================================================
    # KPI STRIP
    # =============================================================================
    total_tasks = len(task_id_map)
    unique_annotators = annotator_task_df["AnnotatorID"].nunique()
    scope_cat = data_csv["scope_category"] if "scope_category" in data_csv.columns else None
    unique_categories = int(scope_cat.nunique()) if scope_cat is not None else 0
    scope_batch = data_csv["scope_batchId"] if "scope_batchId" in data_csv.columns else None
    unique_batches = int(scope_batch.nunique()) if scope_batch is not None else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='rlhf-kpi-value'>{total_tasks}</div>"
            "<div class='rlhf-kpi-label'>Tasks</div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='rlhf-kpi-value'>{unique_annotators}</div>"
            "<div class='rlhf-kpi-label'>Annotators</div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='rlhf-kpi-value'>{unique_categories}</div>"
            "<div class='rlhf-kpi-label'>Categories</div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='rlhf-kpi-value'>{unique_batches}</div>"
            "<div class='rlhf-kpi-label'>Batches</div>",
            unsafe_allow_html=True,
        )

    inspect_tab, csv_tab, delivery_tab, validator_tab = st.tabs(
        [
            "🔍 Inspect Task ID",
            "🧾 View CSV",
            "📦 Delivery Batch Creator",
            "✅ Validator (placeholder)"
        ]
    )
    


    # =============================================================================
    # TAB 1 – INSPECT TASK
    # =============================================================================
    with inspect_tab:
        st.subheader("Inspect a Single Task")

        selected_task_id = st.selectbox(
            "Select a Task ID", list(task_id_map.keys()), key="inspect_task_id_select"
        )

        if selected_task_id:
            task = task_id_map[selected_task_id]

            # wider message column
            meta_col, msg_col = st.columns([0.9, 2.1])

            # --- metadata / scope -----------------------------------------------------
            with meta_col:
                st.markdown(f"### 📄 Task `{selected_task_id}`")
                colab_link = task.get("task", {}).get("colabLink", "N/A")
                st.markdown(f"🔗 **RLHF Link**: [Open Task]({colab_link})")

                st.markdown("#### 📘 Scope")
                scope = task.get("metadata", {}).get("scope_requirements", {})
                scope_rows = [{"Key": k, "Value": str(v)} for k, v in scope.items()]

                # annotator
                try:
                    annotator_id_list = annotator_task_df[
                        annotator_task_df["task_id"] == int(selected_task_id)
                    ]["AnnotatorID"].tolist()
                    annotator_id = annotator_id_list[0] if annotator_id_list else "NA"
                except Exception:  # noqa: BLE001
                    annotator_id = "NA"
                scope_rows.append({"Key": "annotator_id", "Value": annotator_id})

                scope_df = pd.DataFrame(scope_rows)
                scope_df.columns = ["Key", "Value"]

                # clickable link for http
                def _mk_link(v):
                    if isinstance(v, str) and v.startswith("http"):
                        return f"[link]({v})"
                    return v

                scope_df["Value"] = scope_df["Value"].apply(_mk_link)

                st.dataframe(
                    scope_df.set_index("Key"),
                    use_container_width=True,
                    height=min(400, 35 * len(scope_df) + 40),
                )

            # --- messages -------------------------------------------------------------
            with msg_col:
                st.markdown("### 💬 Messages")
                messages = task.get("messages", [])
                for i, msg in enumerate(messages, start=1):
                    role_raw = msg.get("role", "")
                    role = role_raw.capitalize()
                    badge_class = (
                        "rlhf-badge-user"
                        if role == "User"
                        else "rlhf-badge-assistant" if role == "Assistant" else ""
                    )

                    with st.expander(f"{role} msg #{i}", expanded=(i == 1)):
                        st.markdown(
                            f"<span class='{badge_class}'>{role}</span>",
                            unsafe_allow_html=True,
                        )
                        if role == "User":
                            prompt_text = msg.get("text", "")
                            st.markdown("**Prompt (text):**")
                            st.code(prompt_text)

                            st.markdown("**Other Fields in User Message**")
                            for key, val in msg.items():
                                if key in ["text", "role"]:
                                    continue
                                st.markdown(f"##### ▸ {key}")
                                if key == "prompt_evaluation" and isinstance(val, list):
                                    rows = [
                                        {
                                            "Question": item.get("question", ""),
                                            "Description": item.get("description", ""),
                                            "Human Answer": item.get(
                                                "human_input_value", ""
                                            ),
                                        }
                                        for item in val
                                    ]
                                    _safe_show_eval_rows(rows)
                                else:
                                    if isinstance(val, (dict, list)):
                                        st.json(val)
                                    else:
                                        st.write(val)

                        elif role == "Assistant":
                            signal = msg.get("signal", {})
                            ideal_response = signal.get("ideal_response")
                            if ideal_response:
                                st.markdown("**💡 Ideal Response:**")
                                st.code(ideal_response)

                            human_evals = signal.get("human_evals", [])
                            for eval_set in human_evals:
                                eval_form = eval_set.get("evaluation_form", [])
                                if eval_form:
                                    st.markdown("**🧾 Human Evaluation Table:**")
                                    rows = [
                                        {
                                            "Question": item.get("question", ""),
                                            "Description": item.get("description", ""),
                                            "Human Answer": item.get(
                                                "human_input_value", ""
                                            ),
                                        }
                                        for item in eval_form
                                    ]
                                    _safe_show_eval_rows(rows)


    # =============================================================================
    # TAB 2 – VIEW CSV + FILTERS
    # =============================================================================
    with csv_tab:
        st.subheader("Compiled Task Table")
        st.caption(
            "Filters available for: task_id, scope_category, scope_batchName, scope_batchId."
        )

        if data_csv.empty:
            st.warning("No compiled task rows.")
        else:
            with st.expander("🔍 Add Filters", expanded=False):
                filtered_df = filter_dataframe(data_csv, allowed_cols=ALLOWED_FILTER_COLS)

            if "filtered_df" not in locals():
                filtered_df = data_csv.copy()

            st.markdown(
                f"**Showing {len(filtered_df):,} of {len(data_csv):,} rows.**",
                unsafe_allow_html=True,
            )

            st.dataframe(filtered_df, use_container_width=True)

            # Sticky toolbar
            st.markdown("<div class='rlhf-toolbar-sticky'></div>", unsafe_allow_html=True)

            # Create binary Excel files in memory
            full_buf = io.BytesIO()
            filt_buf = io.BytesIO()

            # Save Excel files to buffers
            with pd.ExcelWriter(full_buf, engine='xlsxwriter') as writer:
                data_csv.to_excel(writer, index=False)
            with pd.ExcelWriter(filt_buf, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False)

            # Rewind the buffer to the beginning
            full_buf.seek(0)
            filt_buf.seek(0)

            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="⬇️ Download FULL Data",
                    data=full_buf,
                    file_name="rlhf_tasks_export_full.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col_dl2:
                st.download_button(
                    label="⬇️ Download FILTERED Data",
                    data=filt_buf,
                    file_name="rlhf_tasks_export_filtered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


    # =============================================================================
    # TAB 3 – DELIVERY BATCH CREATOR
    # =============================================================================
    with delivery_tab:
        st.subheader("Delivery Batch Creator")
        st.caption(
            "Upload an *Ingestion JSON* (work items) below. We'll build a Delivery payload from the compiled tasks."
        )

        ingestion_file = st.file_uploader(
            "Upload Ingestion JSON", type="json", key="ingestion_json_uploader_tab"
        )
        ingestion_data = load_json(ingestion_file)

        if ingestion_file and ingestion_data is None:
            st.stop()

        if ingestion_data is None:
            st.info("Upload an ingestion JSON to enable batch creation.")
        else:
            st.success("Ingestion JSON loaded.")

            final_json_dict = build_delivery_json_from_ingestion(
                ingestion_data=ingestion_data,
                data_csv=data_csv,
                data_sft=data_sft,
                pdf_col="scope_pdf_name",
            )

            if final_json_dict:
                workitems = final_json_dict.get("workitems", [])
                total_wi = len(workitems)
                show_max = 3

                st.markdown(
                    f"<div class='rlhf-delivery-summary'>Total workitems built: "
                    f"<strong>{total_wi}</strong>. Showing first {min(show_max, total_wi)} below.</div>",
                    unsafe_allow_html=True,
                )

                preview_dict = dict(final_json_dict)
                preview_dict["workitems"] = workitems[:show_max]

                expand_full = st.checkbox(
                    "Show full JSON inline (may be large)", value=False, key="show_full_json"
                )
                if expand_full:
                    st.json(final_json_dict)
                else:
                    st.json(preview_dict)

                # download full JSON
                final_json_str = json.dumps(final_json_dict, ensure_ascii=False, indent=2)
                final_json_bytes = final_json_str.encode("utf-8")
                fname = timestamped_json_filename()

                st.download_button(
                    label="📦 Download FULL Final JSON",
                    data=final_json_bytes,
                    file_name=fname,
                    mime="application/json",
                    key="download_final_json_tab",
                )

                # download CSV
                csv_buf = io.StringIO()
                data_csv.to_csv(csv_buf, index=False)
                st.download_button(
                    label="🧾 Download Task CSV",
                    data=csv_buf.getvalue(),
                    file_name="rlhf_tasks_export.csv",
                    mime="text/csv",
                    key="download_task_csv_tab",
                )
            else:
                st.warning("No delivery batch could be built (see warnings/errors above).")


    # =============================================================================
    # TAB 4 – VALIDATOR (placeholder)
    # =============================================================================
    with validator_tab:
        st.subheader("Validator – Coming Soon")
        st.caption("Upload an output schema file for future validation (no logic yet).")
        st.markdown(
            "<div class='rlhf-dropzone'>Drop a schema file below (JSON, CSV, XLSX, YAML). Validation not yet implemented.</div>",
            unsafe_allow_html=True,
        )

        validator_file = st.file_uploader(
            "Upload Output Schema File",
            type=["json", "csv", "xlsx", "yaml", "yml"],
            key="validator_schema_uploader",
        )

        if validator_file is not None:
            st.info("File received. Validation logic not yet implemented.")
        else:
            st.write("No schema uploaded yet.")


# =============================================================================
# JSON VISUALIZER TAB (Standalone)
# =============================================================================
elif option == "JSON Visualizer":
    st.title("📊 JSON Visualizer & Explainer (Powered by GPT-4o)")

    st.markdown("### 📁 Upload Your JSON File")
    json_file = st.file_uploader("Upload JSON File", type="json", key="visualizer_json_uploader")

    if json_file:
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            st.error("❌ OpenAI API key not found in .env file")
            st.stop()

        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        json_data = None
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
        st.info("📥 Upload a JSON file to begin analysis.")
