# Enhanced RLHF & JSON Viewer - Complete Professional Implementation
import streamlit as st
import json
import pandas as pd
import io
import re
import uuid
import numpy as np
from datetime import datetime, timezone
from json_mapper_tab import render_json_mapper_tab
from jsonschema import Draft7Validator, ValidationError
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import copy
from dotenv import load_dotenv
import openai

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
class AppConfig:
    """Centralized configuration for application constants"""
    PAGE_TITLE = "RLHF & JSON Viewer"
    PAGE_ICON = "🔍"
    LAYOUT = "wide"
    ALLOWED_FILTER_COLS = ["task_id", "scope_category", "scope_batchName", "scope_batchId"]
    CUSTOM_DELIVERY_MIN_TURNS = 3
    JSON_VISUALIZER_MODEL = "gpt-4o-2024-05-13"
    JSON_VISUALIZER_MAX_TOKENS = 4096
    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.I
    )

# =============================================================================
# CUSTOM CSS (responsive width + visual polish)
# =============================================================================
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE, 
    layout=AppConfig.LAYOUT,
    page_icon=AppConfig.PAGE_ICON
)

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

/* loading spinner */
.rlhf-spinner-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100px;
}

/* custom deliverable builder */
.rlhf-custom-builder-section {
  border: 1px solid var(--rlhf-gray-border);
  border-radius: var(--rlhf-card-radius);
  padding: var(--rlhf-card-pad);
  margin-bottom: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_json(uploaded_file: Optional[io.BytesIO]) -> Optional[Union[dict, list]]:
    """Safely load JSON from uploaded file with error handling"""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        return json.load(uploaded_file)
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON: Invalid JSON syntax at line {e.lineno}")
    except Exception as e:
        st.error(f"❌ Failed to read JSON file: {str(e)}")
    return None

def current_utc_iso_millis() -> str:
    """Get current UTC time in ISO format with milliseconds"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def timestamped_json_filename(prefix: str = "rlhf_final_payload", suffix: str = ".json") -> str:
    """Generate a timestamped filename for JSON exports"""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}{suffix}"

def _clean_col(name: str) -> str:
    """Make a safe Excel/CSV column name while preserving original wording."""
    name = re.sub(r"[\\/\[\]:*?\n\r]", " ", name).strip()
    return re.sub(r"\s{2,}", " ", name)

def _jsonify(val: Any) -> str:
    """Convert dict/list objects to compact JSON strings for Excel compatibility"""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return val

def _safe_key(s: str) -> str:
    """Convert string to safe key format for Streamlit component keys"""
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)

# =============================================================================
# CORE FUNCTIONALITY
# =============================================================================
class RLHFProcessor:
    """Main processor for RLHF data handling and transformation"""
    @staticmethod
    def build_annotator_task_df(data_sft: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build dataframe mapping annotators to their tasks"""
        annotator_task_dict = {}
        for task in data_sft:
            task_id = task.get("id")
            annotator_id = task.get("humanRoleId")
            if task_id and annotator_id:
                annotator_task_dict.setdefault(annotator_id, []).append(task_id)

        return (
            pd.Series(annotator_task_dict)
            .explode()
            .reset_index(name="task_id")
            .rename(columns={"index": "AnnotatorID"})
        )

    @staticmethod
    def compile_task_rows(
        task_id_map: Dict[str, Dict[str, Any]],
        data_sft: List[Dict[str, Any]],
        annotator_task_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compile all task data into a structured dataframe"""
        rows: List[Dict[str, Any]] = []

        for task_id, task in task_id_map.items():
            base = {
                "task_id": task_id,
                "colab_link": task.get("task", {}).get("colabLink", "")
            }

            # Add scope requirements
            scope = task.get("metadata", {}).get("scope_requirements", {})
            for k, v in scope.items():
                base[f"scope_{k}"] = v

            # System message
            sys_msg = next(
                (m for m in task.get("messages", []) if m.get("role") == "system"), 
                None
            )
            if sys_msg:
                base["system_message"] = sys_msg.get("text", "")

            # Process messages
            users = [m for m in task.get("messages", []) if m.get("role") == "user"]
            assistants = [m for m in task.get("messages", []) if m.get("role") == "assistant"]
            n_turns = max(len(users), len(assistants))

            for idx in range(n_turns):
                row = base.copy()
                row["turn_number"] = idx + 1

                # User message
                if idx < len(users):
                    user_msg = users[idx]
                    row["prompt"] = user_msg.get("text", "")
                    for pe in user_msg.get("prompt_evaluation", []):
                        col = _clean_col(pe.get("question", ""))
                        row[col] = pe.get("human_input_value", "")
                    for k, v in user_msg.items():
                        if k not in {"role", "text", "prompt_evaluation"}:
                            row[f"user_{k}"] = _jsonify(v)
                else:
                    row["prompt"] = ""

                # Assistant message
                if idx < len(assistants):
                    assistant_msg = assistants[idx]
                    signal = assistant_msg.get("signal", {})
                    opts = assistant_msg.get("response_options", [])
                    row["model_response"] = opts[0].get("text", "") if opts else ""
                    row["ideal_response"] = signal.get("ideal_response", "")
                    for he_set in signal.get("human_evals", []):
                        for item in he_set.get("evaluation_form", []):
                            col = _clean_col(item.get("question", ""))
                            row[col] = item.get("human_input_value", "")
                else:
                    row["model_response"] = ""
                    row["ideal_response"] = ""

                row.update({
                    "total_user_messages": len(users),
                    "total_assistant_messages": len(assistants),
                    "total_turns": n_turns
                })
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["task_id_orig"] = df["task_id"]
        df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")
        return df.merge(annotator_task_df, on="task_id", how="left")

    @staticmethod
    def create_excel_export(df: pd.DataFrame) -> io.BytesIO:
        """Create Excel file in memory from dataframe"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return output

    @staticmethod
    def row_to_record(row: pd.Series, eval_cols: List[str]) -> Dict[str, Any]:
        """Convert a dataframe row to delivery record format"""
        meta = {col: row[col] for col in eval_cols if col in row.index}
        meta = {k: (None if pd.isna(v) else v) for k, v in meta.items()}
        question = row["prompt"] if "prompt" in row and not pd.isna(row["prompt"]) else None
        answer = row["ideal_response"] if "ideal_response" in row and not pd.isna(row["ideal_response"]) else None
        if "system_message" in eval_cols:
            system_message = row["system_message"] if "system_message" in row and not pd.isna(row["system_message"]) else None
            return {"system_message": system_message, "question": question, "answer": answer, "metadata": meta}
        return {"question": question, "answer": answer, "metadata": meta}

    @staticmethod
    def build_delivery_json_from_ingestion(
        ingestion_data: Dict[str, Any],
        data_csv: pd.DataFrame,
        data_sft: List[Dict[str, Any]],
        pdf_col: str = "scope_pdf_name",
        min_turns: int = AppConfig.CUSTOM_DELIVERY_MIN_TURNS
    ) -> Dict[str, Any]:
        """Build delivery JSON from ingestion format"""
        if not ingestion_data:
            return {}

        sft_annotation = []
        for task in data_sft:
            subset = {k: task.get(k) for k in ("createdAt", "id", "humanRoleId")}
            sft_annotation.append(subset)
        sft_annotation_df = pd.DataFrame(sft_annotation)

        if pdf_col not in data_csv.columns:
            st.warning(f"Column `{pdf_col}` not found; using single dummy group for delivery batch build.")
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
        for index, work_id in enumerate(work_item_ids_needed):
            if index >= len(pdf_unique):
                break
            pdf_name = pdf_unique[index]
            data_tmp = data_csv.loc[data_csv[pdf_col] == pdf_name].reset_index(drop=True)
            if data_tmp.shape[0] < min_turns:
                skipped_groups.append((pdf_name, data_tmp.shape[0]))
                continue

            annot_vals = data_tmp["AnnotatorID"].dropna().unique().tolist() if "AnnotatorID" in data_tmp else []
            annotator_id = str(annot_vals[0]) if annot_vals else "NA"

            records = [RLHFProcessor.row_to_record(r, eval_cols) for _, r in data_tmp.iterrows()]
            task_answers = {"id": pdf_name, "annotations": records}
            task_answers_dict = {"taskAnswers": [task_answers]}

            variation_task_ids = data_tmp["task_id"].unique().tolist()
            try:
                labelled_ts = sft_annotation_df[sft_annotation_df["id"].isin(variation_task_ids)]["createdAt"].tolist()[-1]
            except Exception:
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
            st.warning(
                f"Skipped {len(skipped_groups)} groups with fewer than {min_turns} turns: "
                + ", ".join([f"{name}({n})" for name, n in skipped_groups[:5]])
                + ("..." if len(skipped_groups) > 5 else "")
            )

        return {
            "fileMetadata": ingestion_data.get("fileMetadata"),
            "workitems": master_list,
        }

class JSONValidator:
    """Handles JSON schema validation functionality"""
    @staticmethod
    def _format_validation_error(err: ValidationError, item_idx: int, workitem_id: str) -> Dict[str, Any]:
        """Format schema validation error for display"""
        path = " → ".join(map(str, err.path))
        error_messages = {
            "required": f"Missing required field(s): {', '.join(err.schema['required'])}",
            "maxItems": f"Too many items (max {err.validator_value} allowed)",
            "type": f"Invalid type (expected {err.validator_value})",
            "pattern": f"Value doesn't match required pattern: {err.validator_value}",
            "oneOf": "Value must match one of the specified schemas",
            "enum": f"Value must be one of: {', '.join(map(str, err.validator_value))}"
        }
        brief = error_messages.get(err.validator, err.message)
        return {
            "workItemId": workitem_id,
            "item_index": item_idx,
            "error_path": path,
            "brief_message": brief,
            "validator": err.validator,
            "raw_message": err.message,
        }

    @staticmethod
    def validate_delivery(delivery_data: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate delivery JSON against schema and return errors"""
        if not delivery_data or not schema:
            raise ValueError("Both delivery data and schema must be provided")
        if "workitems" not in delivery_data or not isinstance(delivery_data["workitems"], list):
            raise ValueError("Delivery JSON must contain a 'workitems' list")

        validator = Draft7Validator(schema)
        errors = []
        for idx, workitem in enumerate(delivery_data["workitems"]):
            workitem_id = workitem.get("workItemId", f"INDEX_{idx}")
            for validation_error in validator.iter_errors(workitem):
                errors.append(JSONValidator._format_validation_error(validation_error, idx, workitem_id))
        return errors

class CustomDeliverableBuilder:
    """Handles the custom deliverable builder functionality (robust schema support + optional index expansion)."""

    # ------------------------------ Basic IO & Coercion ------------------------------
    @staticmethod
    def _load_json(uploaded_file: Optional[io.BytesIO]) -> Optional[Union[dict, list]]:
        return load_json(uploaded_file)

    @staticmethod
    def _default_for(jtype: str) -> Any:
        return {
            "object": {},
            "array": [],
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "string": ""
        }.get(jtype, "")
    
    @staticmethod
    def _normalize_override(ov: Dict[str, Any]) -> Dict[str, Any]:
        ov = ov or {}
        return {
            "kind": ov.get("kind", "schema"),
            "col": ov.get("col", ""),
            "pairs": ov.get("pairs", []),
            "group_key": ov.get("group_key", ""),
            "top_key": ov.get("top_key", "items"),
            "children": ov.get("children", {}),   # for dict[nested]
            "item": ov.get("item", None),         # for list[nested]
            "exact_group_size": ov.get("exact_group_size", None),
            "min_group_size": ov.get("min_group_size", None)
        }
    
    @staticmethod
    def _kind_default(kind: str) -> Any:
        base = {
            "schema": "",
            "string": "",
            "dict": {},
            "list[string]": [],
            "list[dict]": [],
            "dict[list[dict]]": {"items": []},
            "dict[nested]": {},
            "list[nested]": []
        }
        return base.get(kind, "")

    @staticmethod
    def _coerce(val: Any, jtype: str) -> Any:
        if pd.isna(val) or val in ("", None):
            return CustomDeliverableBuilder._default_for(jtype)
        try:
            if jtype == "string":
                return str(val)
            if jtype == "integer":
                return int(float(val))
            if jtype == "number":
                return float(val)
            if jtype == "boolean":
                if isinstance(val, bool):
                    return val
                return str(val).strip().lower() in ("true", "1", "yes")
            if jtype == "array":
                if isinstance(val, (list, dict)):
                    return val
                return [v.strip() for v in str(val).split(",") if v.strip()]
            if jtype == "object":
                if isinstance(val, (dict, list)):
                    return val
                try:
                    return json.loads(val)
                except Exception:
                    return {}
        except Exception:
            return CustomDeliverableBuilder._default_for(jtype)
        return val

    # ------------------------------ Schema Helpers ------------------------------
    @staticmethod
    def _materialize_struct_value(
        path: str,
        kind: str,
        cfg: Dict[str, Any],
        this_row: Dict[str, Any],
        all_df: pd.DataFrame,
        typemap: Dict[str, Union[str, List[str]]]
    ) -> Any:
        if kind == "schema":
            col = cfg.get("col", "")
            if not col:
                return "__LEAVE_SKELETON__"
            jtype = typemap.get(path, "string")
            jtype = jtype[0] if isinstance(jtype, list) else jtype
            return CustomDeliverableBuilder._coerce(
                CustomDeliverableBuilder._val_from_col(this_row, col),
                str(jtype)
            )

        if kind == "string":
            col = cfg.get("col", "")
            return (
                CustomDeliverableBuilder._val_from_col(this_row, col)
                if col else "__LEAVE_SKELETON__"
            )

        if kind == "dict":
            pairs = cfg.get("pairs", [])
            if not pairs:
                return {}
            out = {}
            for pair in pairs:
                k = (pair.get("key") or "").strip()
                col = pair.get("col", "")
                if k:
                    out[k] = (
                        CustomDeliverableBuilder._val_from_col(this_row, col)
                        if col else ""
                    )
            return out

        exact_group_size = cfg.get("exact_group_size")

        if kind == "list[string]":
            gk, col = cfg.get("group_key", ""), cfg.get("col", "")
            if not gk or not col:
                return []
            ref = this_row
            group_members = [
                rd for _, r in all_df.iterrows()
                for rd in [r.to_dict()]
                if CustomDeliverableBuilder._row_matches_group(rd, ref, gk)
            ]
            if exact_group_size is None or len(group_members) == exact_group_size:
                items = []
                for rd in group_members:
                    v = CustomDeliverableBuilder._val_from_col(rd, col)
                    if v != "":
                        items.append(v)
                return items
            return []

        if kind == "list[dict]":
            gk, pairs = cfg.get("group_key", ""), cfg.get("pairs", [])
            if not gk or not pairs:
                return []
            ref = this_row
            group_members = [
                rd for _, r in all_df.iterrows()
                for rd in [r.to_dict()]
                if CustomDeliverableBuilder._row_matches_group(rd, ref, gk)
            ]
            if exact_group_size is None or len(group_members) == exact_group_size:
                items = []
                for rd in group_members:
                    d = {}
                    for pair in pairs:
                        k = (pair.get("key") or "").strip()
                        col = pair.get("col", "")
                        if k:
                            d[k] = (
                                CustomDeliverableBuilder._val_from_col(rd, col)
                                if col else ""
                            )
                    items.append(d)
                return items
            return []

        if kind == "dict[list[dict]]":
            top_key = (cfg.get("top_key") or "items").strip() or "items"
            gk, pairs = cfg.get("group_key", ""), cfg.get("pairs", [])
            if not gk or not pairs:
                return {top_key: []}
            ref = this_row
            group_members = [
                rd for _, r in all_df.iterrows()
                for rd in [r.to_dict()]
                if CustomDeliverableBuilder._row_matches_group(rd, ref, gk)
            ]
            if exact_group_size is None or len(group_members) == exact_group_size:
                items = []
                for rd in group_members:
                    d = {}
                    for pair in pairs:
                        k = (pair.get("key") or "").strip()
                        col = pair.get("col", "")
                        if k:
                            d[k] = (
                                CustomDeliverableBuilder._val_from_col(rd, col)
                                if col else ""
                            )
                    items.append(d)
                return {top_key: items}
            return {top_key: []}

        # Fallback for unknown kinds
        return "__LEAVE_SKELETON__"


    @staticmethod
    def _unwrap_schema_root(s: dict) -> dict:
        """Unwrap common wrappers or nested OpenAPI-style components."""
        for k in ("outputSchema", "schema"):
            if isinstance(s, dict) and isinstance(s.get(k), dict):
                return s[k]
        if isinstance(s, dict) and ("properties" in s or "type" in s or "$schema" in s):
            return s
        node = s
        for k in ("components", "schemas"):
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                node = None
                break
        if isinstance(node, dict) and node:
            first = next((v for v in node.values() if isinstance(v, dict)), None)
            if first:
                return first
        return s

    @staticmethod
    def _collapse_duplicate_index_branches(
        leaves: List[str],
        typemap: Dict[str, Union[str, List[str]]]
    ) -> Tuple[List[str], Dict[str, Union[str, List[str]]]]:
        """
        If a parent array has multiple indices (e.g., .0 and .1) with identical child key sets,
        keep only the first index's paths and drop the duplicates.
        """
        import re
        from collections import defaultdict

        by_parent: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        idx_re = re.compile(r"^(.*)\.(\d+)\.(.+)$")

        for p in leaves:
            m = idx_re.match(p)
            if not m:
                continue
            parent, idx, rest = m.groups()
            by_parent[parent][idx].add(rest)

        to_remove = set()
        for parent, idx_map in by_parent.items():
            seen_signatures: Dict[tuple, str] = {}
            for idx, child_set in sorted(idx_map.items(), key=lambda kv: int(kv[0])):
                sig = tuple(sorted(child_set))
                if sig in seen_signatures:
                    prefix = f"{parent}.{idx}."
                    to_remove.update([p for p in leaves if p.startswith(prefix)])
                else:
                    seen_signatures[sig] = idx

        pruned = [p for p in leaves if p not in to_remove]
        for p in to_remove:
            typemap.pop(p, None)
        return sorted(pruned), typemap

    @staticmethod
    def _canonicalize_schema_node(node: Any) -> str:
        """Stable signature for a schema subnode (drop non-structural metadata)."""
        DROP = {"title", "description", "examples"}
        def strip_meta(obj):
            if isinstance(obj, dict):
                return {k: strip_meta(v) for k, v in obj.items() if k not in DROP}
            if isinstance(obj, list):
                return [strip_meta(x) for x in obj]
            return obj
        try:
            return json.dumps(strip_meta(node), sort_keys=True)
        except Exception:
            return str(type(node))

    @staticmethod
    def gather_schema_paths(
        node: Any,
        path: str = "",
        seen: Optional[set] = None,
        bag: Optional[set] = None,
        tmap: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> Tuple[List[str], Dict[str, Union[str, List[str]]]]:
        """
        Exhaustive, de-duplicated schema path extractor:
        - walks properties, items, and oneOf/anyOf/allOf (no early returns)
        - enumerates tuple-typed arrays (.0, .1, ...) but SKIPS structurally-identical entries
        - exemplar .0 for homogeneous arrays (items: {...})
        - dedupes by path; merges types across variants
        """
        if seen is None:
            seen = set()
        if bag is None:
            bag = set()
        if tmap is None:
            tmap = {}

        # Avoid re-walking identical dict nodes
        if isinstance(node, dict):
            nid = id(node)
            if nid in seen:
                return sorted(bag), tmap
            seen.add(nid)

        def _merge_type(p: str, t: Optional[Union[str, List[str]]]):
            if not p:
                return
            if t is None:
                types = ["string"]
            elif isinstance(t, list):
                types = [str(x) for x in t if x] or ["string"]
            else:
                types = [str(t)]
            if p not in tmap:
                tmap[p] = types[0] if len(types) == 1 else sorted(set(types))
            else:
                prev = tmap[p] if isinstance(tmap[p], list) else [tmap[p]]
                merged = sorted(set(prev).union(types))
                tmap[p] = merged[0] if len(merged) == 1 else merged
            bag.add(p)

        # ---- dict node
        if isinstance(node, dict):
            declared_type = node.get("type")
            if declared_type in ("object", "array", "string", "number", "integer", "boolean"):
                _merge_type(path, declared_type)
            elif "properties" in node or "items" in node:
                _merge_type(path, "object" if "properties" in node else "array")

            # 1) properties
            props = node.get("properties")
            if isinstance(props, dict) and props:
                for k, v in props.items():
                    new_path = f"{path}.{k}" if path else k
                    CustomDeliverableBuilder.gather_schema_paths(v, new_path, seen, bag, tmap)

            # 2) items
            if "items" in node:
                items = node["items"]
                if isinstance(items, list) and items:
                    # tuple-typed array → de-dupe identical entries by structure
                    sig_to_first_idx: Dict[str, int] = {}
                    for i, sub in enumerate(items):
                        sig = CustomDeliverableBuilder._canonicalize_schema_node(sub)
                        if sig in sig_to_first_idx:
                            continue
                        sig_to_first_idx[sig] = i
                        idx_path = f"{path}.{i}" if path else str(i)
                        CustomDeliverableBuilder.gather_schema_paths(sub, idx_path, seen, bag, tmap)
                else:
                    # homogeneous → exemplar at .0
                    idx_path = f"{path}.0" if path else "0"
                    CustomDeliverableBuilder.gather_schema_paths(items, idx_path, seen, bag, tmap)

            # 3) union combinators
            for comb in ("oneOf", "anyOf", "allOf"):
                opts = node.get(comb)
                if isinstance(opts, list) and opts:
                    for sub in opts:
                        CustomDeliverableBuilder.gather_schema_paths(sub, path, seen, bag, tmap)

            return sorted(bag), tmap

        # ---- list node (defensive)
        if isinstance(node, list):
            for i, sub in enumerate(node):
                idx_path = f"{path}.{i}" if path else str(i)
                CustomDeliverableBuilder.gather_schema_paths(sub, idx_path, seen, bag, tmap)
            return sorted(bag), tmap

        # ---- primitives / unknowns
        return sorted(bag), tmap

    # ------------------------------ Index Expansion (UI-time) ------------------------------
    @staticmethod
    def _candidate_array_parents(leaves: List[str]) -> List[str]:
        """
        For any path that contains '.0', collect the parent prefix before '.0'.
        e.g., '...taskAnswers.0.answer' -> parent '...taskAnswers'
        """
        parents = set()
        for p in leaves:
            i = p.find(".0")
            if i != -1:
                parents.add(p[:i])
        return sorted(parents)

    @staticmethod
    def _expand_indices(
        leaves: List[str],
        typemap: Dict[str, Union[str, List[str]]],
        parents_to_expand: List[str],
        count: int,
    ) -> Tuple[List[str], Dict[str, Union[str, List[str]]]]:
        """
        Virtually clone exemplar '.0' leaves under selected parents to .1..(count-1).
        Types are copied into typemap for the cloned paths.
        """
        out = set(leaves)
        for p in list(leaves):
            for parent in parents_to_expand:
                anchor = parent + ".0"
                if p.startswith(anchor):
                    for i in range(1, count):
                        q = p.replace(anchor, f"{parent}.{i}", 1)
                        out.add(q)
                        if p in typemap:
                            typemap[q] = typemap[p]
        return sorted(out), typemap

    # ------------------------------ Path Setting & Mapping helpers ------------------------------
    @staticmethod
    def _prune_array_item_exemplars(
        leaves: list[str],
        typemap: dict[str, str | list[str]],
        *,
        drop_scalar_items: bool = True,
        drop_object_item_containers: bool = True,
    ) -> tuple[list[str], dict[str, str | list[str]]]:
        """
        - If parent array path exists (type 'array') and the item exemplar path ends with '.0':
        - When the item's type is a scalar (string/number/integer/boolean), drop the '.0' path.
        - When the item is an object, optionally drop the '.0' container node itself but keep its children.
        """
        def base_type(t):
            return t[0] if isinstance(t, list) and t else (t or "string")

        leafset = set(leaves)

        # 1) Drop scalar exemplars like ...image_url.0 when ...image_url exists
        if drop_scalar_items:
            to_remove = []
            for p in leafset:
                if not p.endswith(".0"):
                    continue
                parent = p[:-2]
                if parent in leafset and base_type(typemap.get(parent)) == "array":
                    item_t = base_type(typemap.get(p))
                    if item_t in ("string", "number", "integer", "boolean"):
                        to_remove.append(p)
            for p in to_remove:
                leafset.discard(p)
                typemap.pop(p, None)

        # 2) Drop object item containers like ...turnLevelOutput.0 (but keep their children)
        if drop_object_item_containers:
            to_remove = []
            for p in list(leafset):
                if not p.endswith(".0"):
                    continue
                has_children = any(x.startswith(p + ".") for x in leafset)
                if has_children:
                    item_t = base_type(typemap.get(p))
                    if item_t in ("object",):
                        to_remove.append(p)
            for p in to_remove:
                leafset.discard(p)
                typemap.pop(p, None)

        return sorted(leafset), typemap

    @staticmethod
    def set_by_path(root: Any, path: str, value: Any) -> None:
        """
        Robustly set a value at dotted path, auto-coercing intermediate containers:
        - If we need a list but find a dict/scalar → replace in parent with []
        - If we need a dict but find a list/scalar → replace in parent with {}
        This prevents collisions when a parent was previously set to a scalar.
        """
        parts = path.split(".")
        current = root
        parent = None
        parent_key = None
        parent_is_index = False

        def _install_in_parent(new_container):
            nonlocal current
            if parent is None:
                # Root should be a dict; if not, reset it safely
                if isinstance(root, dict):
                    # reset root in-place
                    root.clear()
                    root.update(new_container if isinstance(new_container, dict) else {})
                    current = root if isinstance(new_container, dict) else new_container
                else:
                    # extremely rare; just point current to the new container
                    current = new_container
            else:
                if parent_is_index:
                    while len(parent) <= parent_key:
                        parent.append(None)
                    parent[parent_key] = new_container
                else:
                    parent[parent_key] = new_container
                current = new_container

        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            is_index = part.isdigit()
            key = int(part) if is_index else part

            # What container is needed for the *next* hop?
            next_container = [] if (not is_last and parts[i + 1].isdigit()) else {}

            if is_index:
                # Ensure current is a list
                if not isinstance(current, list):
                    _install_in_parent([])
                # current is a list now
                while len(current) <= key:
                    current.append(copy.deepcopy(next_container))
                if is_last:
                    current[key] = value
                    return
                if not isinstance(current[key], (list, dict)):
                    current[key] = copy.deepcopy(next_container)
                parent, parent_key, parent_is_index = current, key, True
                current = current[key]
            else:
                # Ensure current is a dict
                if not isinstance(current, dict):
                    _install_in_parent({})
                # current is a dict now
                if is_last:
                    current[key] = value
                    return
                if key not in current or not isinstance(current[key], (list, dict)):
                    current[key] = copy.deepcopy(next_container)
                parent, parent_key, parent_is_index = current, key, False
                current = current[key]


    @staticmethod
    def _placeholder_for(col: str) -> str:
        return f"{{{{{col}}}}}" if col else ""

    @staticmethod
    def _type_default(jtype: Union[str, List[str]]) -> Any:
        def _d(t: str) -> Any:
            return {
                "object": {},
                "array": [],
                "integer": 0,
                "number": 0.0,
                "boolean": False,
                "string": ""
            }.get(t, "")
        if isinstance(jtype, list):
            # Prefer containers first to keep parents expandable
            for t in ("object", "array"):
                if t in jtype:
                    return _d(t)
            for t in ("string", "number", "integer", "boolean"):
                if t in jtype:
                    return _d(t)
            return _d(jtype[0]) if jtype else ""
        return _d(jtype)


    @staticmethod
    def _get_override(path: str) -> Dict[str, Any]:
        return st.session_state["ds_overrides"].get(path, {
            "kind": "schema",
            "col": "",
            "pairs": [],
            "group_key": "",
            "top_key": "items",
            "children": {},   # <-- NEW
            "item": None      # <-- NEW
        })

    @staticmethod
    def _set_override(path: str, cfg: Dict[str, Any]) -> None:
        st.session_state["ds_overrides"][path] = cfg

    @staticmethod
    def _row_matches_group(df_row: Dict[str, Any], ref_row: Dict[str, Any], group_key: str) -> bool:
        a = df_row.get(group_key)
        b = ref_row.get(group_key)
        try:
            if pd.isna(a) and pd.isna(b):
                return True
        except Exception:
            pass
        return a == b

    @staticmethod
    def _val_from_col(row: Dict[str, Any], col: str) -> str:
        v = row.get(col, "")
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else v
    
    @staticmethod
    def _skeleton_for_override(
        path: str,
        ov: Dict[str, Any],
        typemap: Dict[str, Union[str, List[str]]]
    ) -> Any:
        ov = CustomDeliverableBuilder._normalize_override(ov)
        kind = ov["kind"]

        # simple kinds mirror the existing preview logic
        if kind == "schema":
            col = ov.get("col", "")
            return CustomDeliverableBuilder._placeholder_for(col) if col else CustomDeliverableBuilder._type_default(typemap.get(path, "string"))

        if kind == "string":
            col = ov.get("col", "")
            return CustomDeliverableBuilder._placeholder_for(col) if col else ""

        if kind == "dict":
            ex = {}
            for pair in ov.get("pairs", []):
                k = (pair.get("key") or "").strip()
                col = pair.get("col", "")
                if k:
                    ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
            return ex

        if kind == "list[string]":
            col = ov.get("col", "")
            return [CustomDeliverableBuilder._placeholder_for(col) if col else "", "..."]

        if kind == "list[dict]":
            ex = {}
            for pair in ov.get("pairs", []):
                k = (pair.get("key") or "").strip()
                col = pair.get("col", "")
                if k:
                    ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
            return [ex, "..."]

        if kind == "dict[list[dict]]":
            top_key = (ov.get("top_key") or "items").strip() or "items"
            ex = {}
            for pair in ov.get("pairs", []):
                k = (pair.get("key") or "").strip()
                col = pair.get("col", "")
                if k:
                    ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
            return {top_key: [ex, "..."]}

        # nested kinds
        if kind == "dict[nested]":
            out = {}
            for child_key, child_ov in ov.get("children", {}).items():
                out[child_key] = CustomDeliverableBuilder._skeleton_for_override(f"{path}.{child_key}", child_ov, typemap)
            return out

        if kind == "list[nested]":
            item_ov = ov.get("item") or {"kind": "string", "col": ""}
            ex = CustomDeliverableBuilder._skeleton_for_override(f"{path}[]", item_ov, typemap)
            return [ex, "..."]

        return CustomDeliverableBuilder._type_default(typemap.get(path, "string"))
    
    @staticmethod
    def _edit_override_ui(path: str, ov: Dict[str, Any], csv_cols: List[str]) -> Dict[str, Any]:
        ov = CustomDeliverableBuilder._normalize_override(ov)
        kind = ov["kind"]

        # Common helpers
        def _select_value_col(label_key: str):
            current_col = ov.get("col", "")
            idx = csv_cols.index(current_col) if current_col in csv_cols else 0
            ov["col"] = st.selectbox(label_key, options=csv_cols, index=idx, key=f"col_{_safe_key(path)}")
            return ov

        # Simple kinds use the existing UI:
        if kind == "schema":
            return _select_value_col("Value column")

        if kind == "string":
            return _select_value_col("Value column")

        if kind in ("list[string]",):
            # group config handled by caller (reuse your existing logic)
            return _select_value_col("Value column")

        if kind in ("dict", "list[dict]", "dict[list[dict]]"):
            # Pairs editor is rendered by caller; nothing extra needed here.
            return ov

        # --- dict[nested] editor ---
        if kind == "dict[nested]":
            st.caption("Define nested keys for this dict.")
            # Add child
            c1, c2 = st.columns([2,1])
            with c1:
                new_key = st.text_input("Add child key", value="", key=f"new_child_key_{_safe_key(path)}")
            with c2:
                if st.button("➕ Add child", key=f"add_child_{_safe_key(path)}") and new_key.strip():
                    if "children" not in ov or not isinstance(ov["children"], dict):
                        ov["children"] = {}
                    if new_key not in ov["children"]:
                        ov["children"][new_key] = {"kind": "string", "col": ""}

            # Render each child
            to_delete = []
            for child_key, child_ov in (ov.get("children") or {}).items():
                with st.expander(f"🔹 {child_key}", expanded=False):
                    # choose kind
                    kind_opts = ["schema", "string", "dict", "list[string]", "list[dict]", "dict[list[dict]]", "dict[nested]", "list[nested]"]
                    ck = child_ov.get("kind", "string")
                    kind_idx = kind_opts.index(ck) if ck in kind_opts else 1
                    new_kind = st.selectbox("Kind", options=kind_opts, index=kind_idx, key=f"kind_{_safe_key(path)}_{_safe_key(child_key)}")
                    if new_kind != ck:
                        child_ov = {"kind": new_kind, "col": "", "pairs": [], "children": {}, "item": None, "group_key": "", "top_key": "items"}

                    # simple config for some kinds:
                    if new_kind in ("schema", "string", "list[string]"):
                        col_list = csv_cols
                    else:
                        col_list = csv_cols  # still used in deeper editors if needed

                    # pairs for dict/list[dict]/dict[list[dict]]
                    if new_kind in ("dict", "list[dict]", "dict[list[dict]]"):
                        st.caption("Key → column pairs")
                        pairs = child_ov.get("pairs", []) or [{"key": "", "col": ""}]
                        edit_df = st.data_editor(
                            pd.DataFrame(pairs),
                            num_rows="dynamic",
                            column_config={
                                "key": st.column_config.TextColumn("Key"),
                                "col": st.column_config.SelectboxColumn("CSV Column", options=col_list)
                            },
                            key=f"pairs_{_safe_key(path)}_{_safe_key(child_key)}"
                        )
                        child_ov["pairs"] = edit_df.to_dict(orient="records")

                        if new_kind == "dict[list[dict]]":
                            child_ov["top_key"] = st.text_input(
                                "Top-level key", value=child_ov.get("top_key", "items"),
                                key=f"topkey_{_safe_key(path)}_{_safe_key(child_key)}"
                            )

                    # value column for schema/string/list[string]
                    if new_kind in ("schema", "string", "list[string]"):
                        sel = st.selectbox(
                            "Value column",
                            options=col_list,
                            index=(col_list.index(child_ov.get("col", "")) if child_ov.get("col", "") in col_list else 0),
                            key=f"col_{_safe_key(path)}_{_safe_key(child_key)}"
                        )
                        child_ov["col"] = sel

                    # nested recursion editors
                    if new_kind in ("dict[nested]", "list[nested]"):
                        # push down to nested editor
                        child_ov = CustomDeliverableBuilder._edit_override_ui(f"{path}.{child_key}", child_ov, csv_cols)

                    # delete option
                    if st.button("🗑️ Remove this child", key=f"del_{_safe_key(path)}_{_safe_key(child_key)}"):
                        to_delete.append(child_key)

                    ov["children"][child_key] = child_ov

            for k in to_delete:
                ov["children"].pop(k, None)
            return ov

        # --- list[nested] editor ---
        if kind == "list[nested]":
            st.caption("Define the item structure for this list.")
            # item override default
            if not ov.get("item"):
                ov["item"] = {"kind": "string", "col": ""}

            # show nested editor for item
            with st.expander("🧱 Item structure", expanded=True):
                ov["item"] = CustomDeliverableBuilder._edit_override_ui(f"{path}[]", ov["item"], csv_cols)

            # grouping config (per-field; master grouping still overrides when enabled)
            st.markdown("**Optional local grouping** (ignored if master grouping is enabled)")
            c1, c2 = st.columns(2)
            with c1:
                ov["group_key"] = st.selectbox(
                    "Group by column (optional)",
                    options=[""] + [c for c in csv_cols if c],
                    index=0,
                    key=f"listnested_group_{_safe_key(path)}"
                )
            with c2:
                ov["min_group_size"] = st.number_input(
                    "Minimum records per group", min_value=1, value=1,
                    key=f"listnested_minsize_{_safe_key(path)}"
                )
            return ov

        return ov


    @staticmethod
    def _materialize_value_recursive(
        path: str,
        ov: Dict[str, Any],
        this_row: Dict[str, Any],
        all_df: pd.DataFrame,
        typemap: Dict[str, Union[str, List[str]]],
        *,
        master_grouping_enabled: bool = False,
        master_group_key: str = "",
        master_min_size: int = 1
    ) -> Any:
        ov = CustomDeliverableBuilder._normalize_override(ov)
        kind = ov["kind"]

        # Reuse existing simple kinds via the old dispatcher
        simple_kinds = {"schema", "string", "dict", "list[string]", "list[dict]", "dict[list[dict]]"}
        if kind in simple_kinds:
            return CustomDeliverableBuilder._materialize_struct_value(
                path, kind, ov, this_row, all_df, typemap
            )

        # --- dict[nested] -----------------------------------------------------------
        if kind == "dict[nested]":
            out = {}
            for child_key, child_ov in ov.get("children", {}).items():
                val = CustomDeliverableBuilder._materialize_value_recursive(
                    f"{path}.{child_key}",
                    child_ov,
                    this_row,
                    all_df,
                    typemap,
                    master_grouping_enabled=master_grouping_enabled,
                    master_group_key=master_group_key,
                    master_min_size=master_min_size
                )
                if val == "__LEAVE_SKELETON__":
                    val = CustomDeliverableBuilder._kind_default(child_ov.get("kind", "string"))
                out[child_key] = val
            return out

        # --- list[nested] -----------------------------------------------------------
        if kind == "list[nested]":
            item_ov = ov.get("item") or {"kind": "string", "col": ""}
            # Decide grouping
            use_master = master_grouping_enabled
            group_key = (master_group_key if use_master else ov.get("group_key", "")).strip()
            min_size = (master_min_size if use_master else ov.get("min_group_size") or 1)
            exact_group_size = ov.get("exact_group_size")

            # If no grouping configured anywhere → single item from current row
            if not group_key:
                val = CustomDeliverableBuilder._materialize_value_recursive(
                    f"{path}[]",
                    item_ov,
                    this_row,
                    all_df,
                    typemap,
                    master_grouping_enabled=master_grouping_enabled,
                    master_group_key=master_group_key,
                    master_min_size=master_min_size
                )
                if val == "__LEAVE_SKELETON__":
                    val = CustomDeliverableBuilder._kind_default(item_ov.get("kind", "string"))
                return [val]

            # Grouped mode
            ref = this_row
            group_members = [
                rd for _, r in all_df.iterrows()
                for rd in [r.to_dict()]
                if CustomDeliverableBuilder._row_matches_group(rd, ref, group_key)
            ]
            if exact_group_size is not None and len(group_members) != exact_group_size:
                return []
            if len(group_members) < (min_size or 1):
                return []

            items = []
            for rd in group_members:
                val = CustomDeliverableBuilder._materialize_value_recursive(
                    f"{path}[]",
                    item_ov,
                    rd,
                    all_df,
                    typemap,
                    master_grouping_enabled=master_grouping_enabled,
                    master_group_key=master_group_key,
                    master_min_size=master_min_size
                )
                if val == "__LEAVE_SKELETON__":
                    val = CustomDeliverableBuilder._kind_default(item_ov.get("kind", "string"))
                items.append(val)
            return items

        return "__LEAVE_SKELETON__"

    # ------------------------------ Main UI ------------------------------
    @staticmethod
    def render_interface(custom_tab) -> None:
        """Render the complete custom deliverable builder interface."""
        with custom_tab:
            st.header("🗂️ Custom Deliverable Builder")

            # Schema upload
            schema_file = st.file_uploader(
                "📄 Upload Output-Schema JSON",
                type="json",
                help="Upload the JSON schema that defines your output structure"
            )
            schema_json = CustomDeliverableBuilder._load_json(schema_file)
            if schema_json is None:
                st.info("Upload a schema to start.")
                st.stop()

            # Ensure CSV exists
            data_csv = st.session_state.get("data_csv", pd.DataFrame())
            if data_csv.empty:
                st.error("CSV empty – load RLHF Viewer first.")
                st.stop()

            # Unwrap & Extract
            schema_root = CustomDeliverableBuilder._unwrap_schema_root(schema_json)
            leaves, typemap = CustomDeliverableBuilder.gather_schema_paths(schema_root)
            leaves, typemap = CustomDeliverableBuilder._collapse_duplicate_index_branches(leaves, typemap)
            leaves, typemap = CustomDeliverableBuilder._prune_array_item_exemplars(
                leaves, typemap,
                drop_scalar_items=True,      # removes ...image_url.0 when ...image_url exists
                drop_object_item_containers=True  # removes ...turnLevelOutput.0 (keeps its children)
            )

            # Schema Debugger
            with st.expander("🧪 Schema Debugger", expanded=False):
                st.write({
                    "paths_found": len(leaves),
                    "has_properties": isinstance(schema_root, dict) and "properties" in schema_root,
                    "root_type": schema_root.get("type") if isinstance(schema_root, dict) else None,
                    "$schema": schema_root.get("$schema") if isinstance(schema_root, dict) else None
                })
                if st.checkbox("Show first 150 paths", value=False, key="dbg_show_paths"):
                    st.code("\n".join(leaves[:150] + (["..."] if len(leaves) > 150 else [])))
                if st.checkbox("Show some (path → type)", value=False, key="dbg_show_types"):
                    sample = {p: typemap.get(p) for p in leaves[:40]}
                    st.json(sample)

            if not leaves:
                st.error("No schema paths found. Please verify the uploaded file is a Draft-7 JSON Schema (not a sample payload).")
                st.stop()

            # Optional: index expansion for homogeneous arrays
            with st.expander("🔁 Optional: Expand homogeneous arrays by index", expanded=False):
                parents = CustomDeliverableBuilder._candidate_array_parents(leaves)
                defaults = [p for p in parents if p.split(".")[-1] in ("taskAnswers", "turnLevelOutput", "question", "details")]
                selected_parents = st.multiselect(
                    "Choose parent arrays (exemplar '.0' → '.0,.1,…')",
                    options=parents,
                    default=defaults
                )
                max_index = st.number_input(
                    "Show indices 0..N-1",
                    min_value=1, value=2, help="Set 2 to show .0 and .1"
                )
                if selected_parents and max_index > 1:
                    leaves, typemap = CustomDeliverableBuilder._expand_indices(leaves, typemap, selected_parents, int(max_index))

            # ---------- Include Keys (checkboxes, all checked by default) ----------
            st.session_state.setdefault("include_keys", {})        # path -> bool
            st.session_state.setdefault("include_filter_text", "") # ui filter

            include_map = st.session_state.include_keys
            # keep only current leaves; default any new ones to True
            include_map = {p: include_map.get(p, True) for p in leaves}
            st.session_state.include_keys = include_map

            import hashlib
            with st.expander("✅ Choose schema keys to include", expanded=True):
                tcol1, tcol2, tcol3, tcol4 = st.columns([1,1,1,2])
                with tcol1:
                    if st.button("Select all (filtered)"):
                        for p in list(include_map.keys()):
                            if st.session_state.include_filter_text.strip().lower() in p.lower():
                                include_map[p] = True
                with tcol2:
                    if st.button("Select none (filtered)"):
                        for p in list(include_map.keys()):
                            if st.session_state.include_filter_text.strip().lower() in p.lower():
                                include_map[p] = False
                with tcol3:
                    if st.button("Invert (filtered)"):
                        for p in list(include_map.keys()):
                            if st.session_state.include_filter_text.strip().lower() in p.lower():
                                include_map[p] = not include_map[p]
                with tcol4:
                    st.session_state.include_filter_text = st.text_input(
                        "Filter keys", value=st.session_state.include_filter_text, placeholder="type to filter…"
                    )

                filt = st.session_state.include_filter_text.strip().lower()
                visible_paths = [p for p in leaves if filt in p.lower()]

                st.caption(f"Showing {len(visible_paths)} of {len(leaves)} keys • "
                           f"Selected: {sum(1 for p,v in include_map.items() if v)}")

                for p in visible_paths:
                    key_stable = "inc_" + hashlib.md5(p.encode("utf-8")).hexdigest()
                    include_map[p] = st.checkbox(p, value=include_map[p], key=key_stable)

            # apply selection
            leaves = [p for p in leaves if st.session_state.include_keys.get(p, True)]
            typemap = {p: typemap[p] for p in leaves if p in typemap}

            # Init other session state
            st.session_state.setdefault("use_custom_struct", False)
            st.session_state.setdefault("ds_overrides", {})
            st.session_state.setdefault("schema_mapping", {})
            st.session_state.setdefault("grouping_enabled", False)
            st.session_state.setdefault("grouping_column", "")
            st.session_state.setdefault("grouping_min_size", 1)

            # Master toggle callback
            def toggle_callback():
                if st.session_state.use_custom_struct:
                    st.session_state.ds_overrides = {
                        p: {"kind": "schema", "col": "", "pairs": [], "group_key": "", "top_key": "items", "children": {}, "item": None}
                        for p in leaves
                    }

            # Toggle
            use_custom = st.toggle(
                "Customize structure (override schema default)",
                value=st.session_state.use_custom_struct,
                help="ON → mapping editor is hidden and full custom UI opens. OFF → use mapping editor.",
                key="use_custom_struct",
                on_change=toggle_callback
            )

            # Available CSV columns
            csv_cols = [""] + sorted(data_csv.columns.tolist())

            # Mode selection
            if use_custom:
                st.success("🧩 **Custom Structure Mode** - Advanced field configuration active")

                # Grouping controls (global)
                with st.expander("🔢 Master Grouping Configuration", expanded=True):
                    st.session_state.grouping_enabled = st.toggle(
                        "Enable record grouping",
                        value=st.session_state.grouping_enabled,
                        help="Group records before building JSON structure"
                    )
                    if st.session_state.grouping_enabled:
                        st.session_state.grouping_column = st.selectbox(
                            "Group by column",
                            options=[c for c in csv_cols if c],
                            index=0,
                            key="grouping_column_select"
                        )
                        st.session_state.grouping_min_size = st.number_input(
                            "Minimum records per group",
                            min_value=1,
                            value=1,
                            help="Only include groups with at least this many records"
                        )

                # Ensure all leaves are initialized
                for p in leaves:
                    if p not in st.session_state.ds_overrides:
                        st.session_state.ds_overrides[p] = {
                            "kind": "schema",
                            "col": "",
                            "pairs": [],
                            "group_key": "",
                            "top_key": "items"
                        }

                st.write(f"**Configuring {len(leaves)} fields:**")

                # UX helpers
                topc1, topc2 = st.columns([1, 1])
                with topc1:
                    expand_all = st.checkbox("Expand all", value=True, key="custom_expand_all")
                with topc2:
                    ft = st.text_input("Filter fields (substring)", value="", key="custom_filter").strip().lower()
                visible = [p for p in leaves if ft in p.lower()] if ft else leaves

                kind_labels = {
                        "schema": "Schema (typed default; or coerce from chosen column)",
                        "string": "string (raw from chosen column)",
                        "dict": "dict (key → column)",
                        "list[string]": "list[string] (group & column)",
                        "list[dict]": "list[dict] (group & pairs)",
                        "dict[list[dict]]": "dict[list[dict]] (top key, group & pairs)",
                        # NEW:
                        "dict[nested]": "dict[nested] (define nested child keys & kinds)",
                        "list[nested]": "list[nested] (define nested item structure; optional grouping)",
                    }

                for p in visible:
                    ov = CustomDeliverableBuilder._get_override(p)
                    with st.expander(f"📝 {p}", expanded=expand_all):
                        current_kind = ov.get("kind", "schema")
                        kind_options = ["schema", "string", "dict", "list[string]", "list[dict]", "dict[list[dict]]", "dict[nested]", "list[nested]"]
                        kind_index = kind_options.index(current_kind) if current_kind in kind_options else 0

                        kind = st.selectbox(
                                "Structure type",
                                options=kind_options,
                                # SAFE formatter to avoid KeyError even if options evolve:
                                format_func=lambda k: kind_labels.get(k, k),
                                index=kind_index,
                                key=f"kind_{_safe_key(p)}"
                            )

                        if kind in ("dict[nested]", "list[nested]"):
                            ov = CustomDeliverableBuilder._edit_override_ui(p, ov, csv_cols)
                            CustomDeliverableBuilder._set_override(p, ov)

                        if kind != current_kind:
                            ov = {"kind": kind, "col": "", "pairs": [], "group_key": "", "top_key": "items"}
                            CustomDeliverableBuilder._set_override(p, ov)

                        # Value columns for Schema & String
                        if kind in ("schema", "string"):
                            current_col = ov.get("col", "")
                            col_index = csv_cols.index(current_col) if current_col in csv_cols else 0
                            ov["col"] = st.selectbox(
                                "Value column",
                                options=csv_cols,
                                index=col_index,
                                key=f"col_{_safe_key(p)}"
                            )
                            CustomDeliverableBuilder._set_override(p, ov)

                        # Grouping for list kinds
                        if kind in ("list[string]", "list[dict]", "dict[list[dict]]"):
                            if st.session_state.grouping_enabled:
                                ov.update({
                                    "group_key": st.session_state.grouping_column,
                                    "min_group_size": st.session_state.grouping_min_size
                                })
                                st.info(f"Using master grouping by: {st.session_state.grouping_column}")
                                if kind == "list[string]":
                                    ov["col"] = st.selectbox(
                                        "Value column",
                                        options=csv_cols,
                                        index=0,
                                        key=f"col_{_safe_key(p)}"
                                    )
                            else:
                                c1, c2 = st.columns(2)
                                with c1:
                                    ov["group_key"] = st.selectbox(
                                        "Group by column",
                                        options=csv_cols,
                                        index=0,
                                        key=f"group_key_{_safe_key(p)}"
                                    )
                                with c2:
                                    if kind == "list[string]":
                                        ov["col"] = st.selectbox(
                                            "Value column",
                                            options=csv_cols,
                                            index=0,
                                            key=f"col_{_safe_key(p)}"
                                        )
                            CustomDeliverableBuilder._set_override(p, ov)

                        # Dict key→column pairs
                        if kind in ("dict", "list[dict]", "dict[list[dict]]"):
                            st.caption("Define key → column pairs for dict elements.")
                            pairs = ov.get("pairs", [])
                            edit_df = st.data_editor(
                                pd.DataFrame(pairs if pairs else [{"key": "", "col": ""}]),
                                num_rows="dynamic",
                                column_config={
                                    "key": st.column_config.TextColumn("Key"),
                                    "col": st.column_config.SelectboxColumn("CSV Column", options=csv_cols)
                                },
                                key=f"pairs_{_safe_key(p)}",
                            )
                            ov["pairs"] = edit_df.to_dict(orient="records")
                            CustomDeliverableBuilder._set_override(p, ov)

                        # Top key for dict[list[dict]]
                        if kind == "dict[list[dict]]":
                            ov["top_key"] = st.text_input(
                                "Top-level dict key (static)",
                                value=ov.get("top_key", "items"),
                                key=f"topkey_{_safe_key(p)}"
                            )
                            CustomDeliverableBuilder._set_override(p, ov)

            else:
                # Simple mapping mode
                st.info("📋 **Simple Mapping Mode** - Basic field mapping active")
                mapping = st.session_state["schema_mapping"]
                tbl_df = pd.DataFrame({"key": leaves, "csv_column": [mapping.get(k, "") for k in leaves]})
                Selectbox = getattr(st.column_config, "SelectboxColumn", None)
                col_cfg = {
                    "key": st.column_config.Column(disabled=True),
                    "csv_column": (
                        Selectbox("Map to CSV (blank = unmapped)", options=csv_cols)
                        if Selectbox else st.column_config.Column("Map to CSV", width="medium")
                    ),
                }
                edited = st.data_editor(
                    tbl_df,
                    column_config=col_cfg,
                    hide_index=True,
                    use_container_width=True,
                    key="mapping_editor"
                )
                for k, v in edited.itertuples(index=False):
                    st.session_state["schema_mapping"][k] = v
                mapping = st.session_state["schema_mapping"]

            # ---------------- Skeleton preview ----------------
            st.markdown("#### 📄 Skeleton Template")
            skeleton: Dict[str, Any] = {}

            if use_custom:
                for p in leaves:
                    ov = CustomDeliverableBuilder._get_override(p)
                    kind = ov.get("kind", "schema")
                    if kind == "schema":
                        col = ov.get("col", "")
                        CustomDeliverableBuilder.set_by_path(
                            skeleton, p,
                            CustomDeliverableBuilder._placeholder_for(col) if col
                            else CustomDeliverableBuilder._type_default(typemap.get(p, "string"))
                        )
                    elif kind == "string":
                        col = ov.get("col", "")
                        CustomDeliverableBuilder.set_by_path(skeleton, p,
                            CustomDeliverableBuilder._placeholder_for(col) if col else "")
                    elif kind == "dict":
                        ex = {}
                        for pair in ov.get("pairs", []):
                            k = (pair.get("key") or "").strip()
                            col = pair.get("col", "")
                            if k:
                                ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
                        CustomDeliverableBuilder.set_by_path(skeleton, p, ex)
                    elif kind == "list[string]":
                        col = ov.get("col", "")
                        CustomDeliverableBuilder.set_by_path(skeleton, p, [
                            CustomDeliverableBuilder._placeholder_for(col) if col else "", "..."
                        ])
                    elif kind == "list[dict]":
                        ex = {}
                        for pair in ov.get("pairs", []):
                            k = (pair.get("key") or "").strip()
                            col = pair.get("col", "")
                            if k:
                                ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
                        CustomDeliverableBuilder.set_by_path(skeleton, p, [ex, "..."])
                    elif kind == "dict[list[dict]]":
                        top_key = (ov.get("top_key") or "items").strip() or "items"
                        ex = {}
                        for pair in ov.get("pairs", []):
                            k = (pair.get("key") or "").strip()
                            col = pair.get("col", "")
                            if k:
                                ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
                        CustomDeliverableBuilder.set_by_path(skeleton, p, {top_key: [ex, "..."]})
                    
                    elif kind == "dict[nested]":
                        ex = CustomDeliverableBuilder._skeleton_for_override(p, ov, typemap)
                        CustomDeliverableBuilder.set_by_path(skeleton, p, ex)

                    elif kind == "list[nested]":
                        ex = CustomDeliverableBuilder._skeleton_for_override(p, ov, typemap)
                        CustomDeliverableBuilder.set_by_path(skeleton, p, ex)
                    else:
                        CustomDeliverableBuilder.set_by_path(
                            skeleton, p, CustomDeliverableBuilder._type_default(typemap.get(p, "string"))
                        )
            else:
                mapping = st.session_state["schema_mapping"]
                for p in leaves:
                    jtype = typemap.get(p, "string")
                    col = mapping.get(p, "")
                    raw_type = jtype[0] if isinstance(jtype, list) else jtype
                    raw = data_csv.iloc[0].get(col, "") if (col and not data_csv.empty) else ""
                    val = CustomDeliverableBuilder._coerce(raw, str(raw_type))
                    CustomDeliverableBuilder.set_by_path(skeleton, p, val if col else CustomDeliverableBuilder._type_default(raw_type))

            st.json(skeleton, expanded=False)

            # ---------------- Generate deliverables ----------------
            if st.button("⚙️ Generate deliverables.json"):
                filled_list = []
                if use_custom:
                    # --- grouped path ---
                    if st.session_state.grouping_enabled:
                        grouped = data_csv.groupby(st.session_state.grouping_column)
                        for name, group in grouped:
                            if len(group) >= st.session_state.grouping_min_size:
                                this = copy.deepcopy(skeleton)

                                # Use a reference row from this group
                                ref_row = group.iloc[0].to_dict()  # <-- ADD this

                                for p in leaves:
                                    ov = CustomDeliverableBuilder._get_override(p)
                                    # was: rd  (undefined)  ❌
                                    val = CustomDeliverableBuilder._materialize_value_recursive(
                                        p, ov,
                                        ref_row,          # <-- FIX: pass ref_row
                                        group,            # all_df for grouping-aware materialization
                                        typemap,
                                        master_grouping_enabled=True,
                                        master_group_key=st.session_state.grouping_column,
                                        master_min_size=st.session_state.grouping_min_size
                                    )
                                    if val != "__LEAVE_SKELETON__":
                                        CustomDeliverableBuilder.set_by_path(this, p, val)

                                filled_list.append(this)

                    # --- non-grouped path ---
                    else:
                        for _, row in data_csv.iterrows():
                            this = copy.deepcopy(skeleton)
                            ref_row = row.to_dict()  # name consistently

                            for p in leaves:
                                ov = CustomDeliverableBuilder._get_override(p)
                                val = CustomDeliverableBuilder._materialize_value_recursive(
                                    p, ov,
                                    ref_row,           # ref row for this item
                                    data_csv,          # all_df
                                    typemap,
                                    master_grouping_enabled=False
                                )
                                if val != "__LEAVE_SKELETON__":
                                    CustomDeliverableBuilder.set_by_path(this, p, val)

                            filled_list.append(this)

                else:
                    mapping = st.session_state["schema_mapping"]
                    for _, row in data_csv.iterrows():
                        this = copy.deepcopy(skeleton)
                        rd = row.to_dict()
                        for p in leaves:
                            col = mapping.get(p, "")
                            jtype = typemap.get(p, "string")
                            base_type = jtype[0] if isinstance(jtype, list) else jtype
                            raw = rd.get(col, "") if col else ""
                            val = CustomDeliverableBuilder._coerce(raw, str(base_type))
                            CustomDeliverableBuilder.set_by_path(this, p, val)
                        filled_list.append(this)

                if filled_list:
                    payload = json.dumps(filled_list, ensure_ascii=False, indent=2).encode()
                    custom_name = st.text_input("Filename for custom deliverables", value="deliverables.json", key="custom_deliverables_name")
                    st.download_button(
                        "⬇️ Download deliverables.json",
                        data=payload,
                        file_name=custom_name,
                        mime="application/json"
                    )
                    st.markdown("#### 🔍 Preview of first generated dict")
                    st.json(filled_list[0], expanded=False)

# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================
class UIComponents:
    """Reusable UI components for the application"""
    @staticmethod
    def pivot_builder(df: pd.DataFrame, title: str = "📊 Build Pivot") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Interactive pivot builder over 'df'. Returns (pivot_df, config).
        - Multi-select rows/columns/values
        - Per-value aggregations
        - Optional totals (margins) and fillna
        - Optional row sorting and Top-N
        - Optional flattening of MultiIndex columns
        """
        st.markdown(f"### {title}")

        if df.empty:
            st.warning("No data to pivot.")
            return pd.DataFrame(), {}

        # ----- Controls -----
        with st.expander("⚙️ Pivot Settings", expanded=True):
            all_cols = df.columns.tolist()

            c_top = st.columns(3)
            with c_top[0]:
                rows = st.multiselect("Rows (index)", options=all_cols, default=[], help="Group rows by these columns")
            with c_top[1]:
                cols = st.multiselect("Columns (pivoted to headings)", options=all_cols, default=[], help="Wide columns")
            with c_top[2]:
                # suggest numeric columns for values first, but allow any
                numeric_candidates = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
                values = st.multiselect(
                    "Values (aggregated)",
                    options=(numeric_candidates + [c for c in all_cols if c not in numeric_candidates]),
                    default=[],
                    help="Pick one or more fields to aggregate"
                )

            # per-value aggregation selection
            agg_map = {}
            agg_choices = ["sum", "mean", "count", "nunique", "min", "max", "median"]
            if values:
                st.caption("Per-value aggregation:")
                for v in values:
                    # default to "sum" if numeric-like, else "count"
                    default_agg = "sum" if pd.api.types.is_numeric_dtype(df[v]) else "count"
                    agg_map[v] = st.selectbox(
                        f"• {v}",
                        options=agg_choices,
                        index=(agg_choices.index(default_agg) if default_agg in agg_choices else 0),
                        key=f"agg_{v}"
                    )

            c2 = st.columns(4)
            with c2[0]:
                show_margins = st.checkbox("Show totals (margins)", value=False)
            with c2[1]:
                margins_name = st.text_input("Totals label", value="All")
            with c2[2]:
                fillna_val = st.text_input("Fill empty cells with", value="")
            with c2[3]:
                flatten_cols = st.checkbox("Flatten MultiIndex columns", value=True)

            # sorting & top-N
            c3 = st.columns(3)
            with c3[0]:
                sort_rows = st.checkbox("Sort rows", value=False)
            with c3[1]:
                top_n = st.number_input("Top N rows (after sort)", min_value=0, value=0, step=1, help="0 = show all")
            with c3[2]:
                sort_desc = st.checkbox("Sort descending", value=True)

            # If flattening, we can sort by any flattened name; if not, let user pick a value to sort by
            sort_by = None
            if sort_rows:
                # We don't know final column labels yet; keep a text box to accept a column label after flatten
                sort_by = st.text_input(
                    "Sort by column label (after flatten, e.g. 'revenue | sum')",
                    value=""
                )

        # ----- Build pivot -----
        cfg = {
            "rows": rows, "cols": cols, "values": values, "agg_map": agg_map,
            "show_margins": show_margins, "margins_name": margins_name,
            "fillna_val": fillna_val, "flatten_cols": flatten_cols,
            "sort_rows": sort_rows, "sort_by": sort_by, "sort_desc": sort_desc, "top_n": top_n
        }

        if not values:
            st.info("Select at least one **Value** to create a pivot.")
            return pd.DataFrame(), cfg

        # map string -> function
        agg_funcs = {
            "sum": np.sum, "mean": np.mean, "count": "count", "nunique": pd.Series.nunique,
            "min": np.min, "max": np.max, "median": np.median
        }
        aggfunc = {k: agg_funcs.get(v, np.sum) for k, v in agg_map.items()} if agg_map else np.sum

        try:
            piv = pd.pivot_table(
                df,
                index=rows if rows else None,
                columns=cols if cols else None,
                values=values,
                aggfunc=aggfunc,
                margins=bool(show_margins),
                margins_name=(margins_name or "All"),
                dropna=False,
                observed=False
            )
        except Exception as e:
            st.error(f"Failed to build pivot: {e}")
            return pd.DataFrame(), cfg

        # Fill NaNs (string box → try numeric else keep as string)
        if fillna_val != "":
            try:
                fill_val_cast = float(fillna_val) if fillna_val.replace(".", "", 1).lstrip("-").isdigit() else fillna_val
            except Exception:
                fill_val_cast = fillna_val
            piv = piv.fillna(fill_val_cast)

        # Flatten columns if requested
        if flatten_cols and isinstance(piv.columns, pd.MultiIndex):
            piv.columns = [" | ".join(map(str, tup)).strip() for tup in piv.columns.to_flat_index()]
        if isinstance(piv.index, pd.MultiIndex):
            piv = piv.reset_index()

        # Sort & Top-N
        if sort_rows:
            if sort_by and sort_by in piv.columns:
                piv = piv.sort_values(by=sort_by, ascending=not sort_desc, kind="mergesort")
            else:
                # fallback: sort by first value column found
                sort_candidates = [c for c in piv.columns if c not in rows]
                if sort_candidates:
                    piv = piv.sort_values(by=sort_candidates[0], ascending=not sort_desc, kind="mergesort")
        if top_n and int(top_n) > 0:
            piv = piv.head(int(top_n))

        # Show preview & downloads
        st.markdown("#### Pivot Preview")
        st.dataframe(piv, use_container_width=True, height=420)

        cdl = st.columns(2)
        with cdl[0]:
            csv_bytes = piv.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="pivot.csv", mime="text/csv")
        with cdl[1]:
            import io as _io
            bio = _io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
                piv.to_excel(w, index=False, sheet_name="pivot")
            bio.seek(0)
            st.download_button("⬇️ Download Excel", data=bio, file_name="pivot.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        return piv, cfg

    @staticmethod
    def adv_filters_active() -> bool:
        """True if any advanced filter has a non-empty value."""
        flts = st.session_state.get("adv_filters", [])
        for f in flts:
            op = f.get("op", "")
            val = f.get("val", None)
            val2 = f.get("val2", None)

            if op == "between":
                if (val not in (None, "")) and (val2 not in (None, "")):
                    return True
            else:
                if isinstance(val, list):
                    if len(val) > 0:
                        return True
                elif val not in (None, ""):
                    return True
        return False

    @staticmethod
    def _infer_col_type(s: pd.Series) -> str:
        """Return one of: 'numeric' | 'datetime' | 'categorical' | 'text'."""
        # numeric?
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().mean() >= 0.8:
            return "numeric"

        # datetime?
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        s_dt = pd.to_datetime(s, errors="coerce", utc=True)
        if s_dt.notna().mean() >= 0.8:
            return "datetime"

        # categorical (low cardinality text)
        nunique = s.dropna().astype(str).nunique()
        if nunique > 0 and nunique <= min(50, max(10, int(0.1 * len(s)))):
            return "categorical"

        return "text"

    # ---------- NEW: + Add filter UI over ALL columns ----------
    @staticmethod
    def filter_dataframe_advanced(df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamic filter builder (+ Add filter) over ALL columns.
        - No default values (inputs start empty)
        - Shows pickable 'possible values' where applicable
        - Filters are ANDed
        """
        if df.empty:
            st.caption("No rows to filter.")
            return df

        st.session_state.setdefault("adv_filters", [])

        # toolbar
        t1, t2, t3 = st.columns([1, 1, 4])
        with t1:
            if st.button("➕ Add filter"):
                default_col = df.columns[0]
                st.session_state.adv_filters.append({
                    "id": str(uuid.uuid4())[:8],
                    "col": default_col,
                    "op": "__auto__",   # will be set based on dtype each render
                    "val": None,        # single value OR list (for pick ops)
                    "val2": None        # for between
                })
        with t2:
            if st.button("🗑️ Clear all"):
                st.session_state.adv_filters = []

        # quick chips
        chips = []
        for f in st.session_state.adv_filters:
            op = f.get("op", "")
            if isinstance(f.get("val"), list):
                vtxt = ", ".join(map(str, f.get("val") or []))
            else:
                vtxt = "" if f.get("val") in (None, "") else str(f.get("val"))
            if f.get("val2") not in (None, ""):
                vtxt += f" .. {f.get('val2')}"
            chips.append(f"<span class='rlhf-chip-amber'>{f.get('col','')} {op} {vtxt}</span>")
        if chips:
            st.markdown("**Active filters:** " + " ".join(chips), unsafe_allow_html=True)

        # render rows
        to_delete = []
        for i, f in enumerate(st.session_state.adv_filters):
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1.6, 1.2, 2.4, 2.4, 0.7])

                # Column
                with c1:
                    col_name = st.selectbox(
                        "Column",
                        options=list(df.columns),
                        index=(list(df.columns).index(f["col"]) if f["col"] in df.columns else 0),
                        key=f"adv_col_{f['id']}"
                    )
                    f["col"] = col_name

                series = df[f["col"]]
                col_type = UIComponents._infer_col_type(series)

                # Operators per dtype (with pick options)
                ops_by_type = {
                        "text":        ["contains", "not contains", "equals", "not equals", "in (pick)"],
                        # ⬇️ add contains ops for categorical
                        "categorical": ["is any of", "is none of", "contains", "not contains"],
                        "numeric":     ["==", "!=", ">", ">=", "<", "<=", "between", "in (pick)"],
                        "datetime":    ["on", "before", "after", "between"],
                    }
                default_op = {
                    "text": "contains",
                    "categorical": "is any of",
                    "numeric": "==",
                    "datetime": "on",
                }[col_type]

                with c2:
                    op_options = ops_by_type[col_type]
                    op_index = op_options.index(f.get("op")) if f.get("op") in op_options else op_options.index(default_op)
                    f["op"] = st.selectbox("Op", options=op_options, index=op_index, key=f"adv_op_{f['id']}")

                # Value widgets — NO DEFAULTS
                with c3:
                    # helper to fetch distinct, nicely formatted values
                    def _distinct_values(s, limit=200):
                        vals = s.dropna().unique().tolist()
                        vals = [str(v) for v in vals]
                        vals = sorted(vals, key=lambda x: (len(x), x))[:limit]
                        return vals

                    if col_type == "text":
                        if f["op"] == "in (pick)":
                            # pick from existing values
                            opts = _distinct_values(series)
                            f["val"] = st.multiselect("Values (pick)", options=opts, default=(f["val"] or []), key=f"adv_val_{f['id']}")
                        else:
                            f["val"] = st.text_input("Value", value=(f["val"] or ""), placeholder="enter text…", key=f"adv_val_{f['id']}")

                   
                    elif col_type == "categorical":
                        if f["op"] in ("contains", "not contains"):
                            # substring match against stringified category values
                            f["val"] = st.text_input(
                                "Substring",
                                value=(f["val"] or ""),
                                placeholder="type to match…",
                                key=f"adv_val_{f['id']}"
                            )
                        else:
                            # pick from distinct values (no defaults)
                            opts = _distinct_values(series)
                            f["val"] = st.multiselect(
                                "Values (pick)",
                                options=opts,
                                default=(f["val"] or []),
                                key=f"adv_val_{f['id']}"
                            )

                    elif col_type == "numeric":
                        if f["op"] == "between":
                            v1 = st.text_input("Min", value=(str(f["val"]) if f.get("val") not in (None, "") else ""), placeholder="e.g. 0.0", key=f"adv_min_{f['id']}")
                            v2 = st.text_input("Max", value=(str(f["val2"]) if f.get("val2") not in (None, "") else ""), placeholder="e.g. 100.0", key=f"adv_max_{f['id']}")
                            f["val"], f["val2"] = v1, v2
                        elif f["op"] == "in (pick)":
                            s_num = pd.to_numeric(series, errors="coerce")
                            opts = [str(x) for x in sorted(s_num.dropna().unique().tolist())[:200]]
                            f["val"] = st.multiselect("Values (pick)", options=opts, default=(f["val"] or []), key=f"adv_val_{f['id']}")
                        else:
                            f["val"] = st.text_input("Value", value=(str(f["val"]) if f.get("val") not in (None, "") else ""), placeholder="e.g. 42 or 3.14", key=f"adv_val_{f['id']}")

                    elif col_type == "datetime":
                        if f["op"] == "between":
                            d1 = st.text_input("Start (YYYY-MM-DD)", value=(f["val"] or ""), placeholder="YYYY-MM-DD", key=f"adv_d1_{f['id']}")
                            d2 = st.text_input("End (YYYY-MM-DD)", value=(f["val2"] or ""), placeholder="YYYY-MM-DD", key=f"adv_d2_{f['id']}")
                            f["val"], f["val2"] = d1, d2
                        else:
                            f["val"] = st.text_input("Date (YYYY-MM-DD)", value=(f["val"] or ""), placeholder="YYYY-MM-DD", key=f"adv_d_{f['id']}")

                # (empty spacer column)
                with c4:
                    st.write("")

                with c5:
                    if st.button("✖", help="Remove", key=f"adv_del_{f['id']}"):
                        to_delete.append(i)

        # delete rows
        for idx in reversed(to_delete):
            st.session_state.adv_filters.pop(idx)

        # ---------- APPLY ----------
        out = df.copy()
        for f in st.session_state.adv_filters:
            col, op = f["col"], f["op"]
            val, val2 = f.get("val"), f.get("val2")
            s = out[col]
            ftype = UIComponents._infer_col_type(s)

            try:
                if ftype == "text":
                    s_str = s.astype(str)
                    if op == "contains":
                        if val in (None, ""):  # skip empty
                            continue
                        mask = s_str.str.contains(str(val), case=False, na=False)
                    elif op == "not contains":
                        if val in (None, ""):
                            continue
                        mask = ~s_str.str.contains(str(val), case=False, na=False)
                    elif op == "equals":
                        if val in (None, ""):
                            continue
                        mask = s_str.str.strip() == str(val).strip()
                    elif op == "not equals":
                        if val in (None, ""):
                            continue
                        mask = s_str.str.strip() != str(val).strip()
                    elif op == "in (pick)":
                        vals = [str(x) for x in (val or [])]
                        mask = s_str.isin(vals) if vals else True
                    else:
                        mask = True

                elif ftype == "categorical":
                    s_cat = s.astype(str)
                    if op == "is any of":
                        vals = [str(x) for x in (val or [])]
                        mask = s_cat.isin(vals) if vals else True
                    elif op == "is none of":
                        vals = [str(x) for x in (val or [])]
                        mask = ~s_cat.isin(vals) if vals else True
                    elif op == "contains":
                        if val in (None, ""):
                            continue
                        mask = s_cat.str.contains(str(val), case=False, na=False)
                    elif op == "not contains":
                        if val in (None, ""):
                            continue
                        mask = ~s_cat.str.contains(str(val), case=False, na=False)
                    else:
                        mask = True

                elif ftype == "numeric":
                    s_num = pd.to_numeric(s, errors="coerce")
                    if op == "between":
                        if val in (None, "") or val2 in (None, ""):
                            continue
                        try:
                            v1, v2 = float(val), float(val2)
                            lo, hi = (min(v1, v2), max(v1, v2))
                            mask = s_num.between(lo, hi)
                        except Exception:
                            continue
                    elif op == "in (pick)":
                        nums = []
                        for t in (val or []):
                            try:
                                nums.append(float(str(t)))
                            except Exception:
                                pass
                        mask = s_num.isin(nums) if nums else True
                    else:
                        if val in (None, ""):
                            continue
                        try:
                            fv = float(val)
                        except Exception:
                            continue
                        if op == "==":   mask = s_num == fv
                        elif op == "!=": mask = s_num != fv
                        elif op == ">":  mask = s_num > fv
                        elif op == ">=": mask = s_num >= fv
                        elif op == "<":  mask = s_num < fv
                        elif op == "<=": mask = s_num <= fv
                        else:            mask = True

                elif ftype == "datetime":
                    s_dt = pd.to_datetime(s, errors="coerce", utc=True)
                    if op == "between":
                        if val in (None, "") or val2 in (None, ""):
                            continue
                        try:
                            d1 = pd.to_datetime(val)
                            d2 = pd.to_datetime(val2)
                            lo, hi = (min(d1, d2), max(d1, d2))
                            mask = (s_dt >= lo) & (s_dt <= hi)
                        except Exception:
                            continue
                    else:
                        if val in (None, ""):
                            continue
                        try:
                            d = pd.to_datetime(val)
                        except Exception:
                            continue
                        if op == "on":     mask = s_dt.dt.date == d.date()
                        elif op == "before": mask = s_dt < d
                        elif op == "after":  mask = s_dt > d
                        else:                mask = True
                else:
                    mask = True

                out = out[mask]

            except Exception:
                # graceful skip on malformed input
                pass

        return out
    
    @staticmethod
    def show_kpi_metrics(metrics: Dict[str, Any]):
        """Display KPI metrics in a 4-column layout"""
        cols = st.columns(4)
        metric_items = [
            ("Tasks", "total_tasks", "rlhf-blue"),
            ("Annotators", "unique_annotators", "rlhf-green"),
            ("Categories", "unique_categories", "rlhf-purple"),
            ("Batches", "unique_batches", "rlhf-amber")
        ]
        for i, (label, key, color) in enumerate(metric_items):
            with cols[i]:
                st.markdown(
                    f"<div class='rlhf-kpi-value'>{metrics.get(key, 0)}</div>"
                    f"<div class='rlhf-kpi-label'>{label}</div>",
                    unsafe_allow_html=True
                )

    @staticmethod
    def filter_dataframe(df: pd.DataFrame, allowed_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Interactive dataframe filtering UI"""
        if allowed_cols is None:
            allowed_cols = AppConfig.ALLOWED_FILTER_COLS

        filtered_df = df.copy()
        cols_present = [c for c in allowed_cols if c in df.columns]
        cols_missing = [c for c in allowed_cols if c not in df.columns]
        if cols_missing:
            st.caption(f"Missing filter columns: {', '.join(cols_missing)}")
        if not cols_present:
            return filtered_df

        n_cols = 2
        filter_cols = st.columns(n_cols)
        selected_filters = []

        for i, col in enumerate(cols_present):
            idx = i % n_cols
            with filter_cols[idx]:
                series = df[col].dropna()
                unique_values = series.unique().tolist()
                value_map = {str(v): v for v in unique_values}
                display_labels = sorted(value_map.keys(), key=lambda x: (len(x), x))
                selected_labels = st.multiselect(
                    f"Filter by {col}",
                    options=display_labels,
                    default=[],
                    key=f"filter_{_safe_key(col)}",
                    help=f"Select values to filter {col} column"
                )
                if selected_labels:
                    selected_values = [value_map[l] for l in selected_labels]
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                    selected_filters.append((col, selected_labels))

        if selected_filters:
            chips_html = "".join(
                f"<span class='rlhf-chip-amber'>{col}: {', '.join(vals)}</span>"
                for col, vals in selected_filters
            )
            st.markdown(f"**Active filters:** {chips_html}", unsafe_allow_html=True)

        return filtered_df

    @staticmethod
    def show_task_inspection(task_data: Dict[str, Any], annotator_id: str):
        """Display detailed task inspection view"""
        meta_col, msg_col = st.columns([0.9, 2.1])
        with meta_col:
            st.markdown(f"### 📄 Task `{task_data.get('taskId', '')}`")
            colab_link = task_data.get("task", {}).get("colabLink", "N/A")
            if colab_link != "N/A":
                st.markdown(f"🔗 **RLHF Link**: [Open Task]({colab_link})")
            else:
                st.markdown("🔗 **RLHF Link**: Not available")
            st.markdown("#### 📘 Scope")
            scope = task_data.get("metadata", {}).get("scope_requirements", {})
            scope_rows = [{"Key": k, "Value": str(v)} for k, v in scope.items()]
            scope_rows.append({"Key": "annotator_id", "Value": annotator_id})
            scope_df = pd.DataFrame(scope_rows)
            scope_df["Value"] = scope_df["Value"].apply(
                lambda v: f"[link]({v})" if isinstance(v, str) and v.startswith("http") else v
            )
            st.dataframe(
                scope_df.set_index("Key"),
                use_container_width=True,
                height=min(400, 35 * len(scope_df) + 40),
            )

        with msg_col:
            st.markdown("### 💬 Conversation")
            messages = task_data.get("messages", [])
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "").capitalize()
                badge_class = (
                    "rlhf-badge-user" if role == "User" 
                    else "rlhf-badge-assistant" if role == "Assistant" 
                    else ""
                )
                with st.expander(f"{role} Message #{i}", expanded=(i == 1)):
                    st.markdown(f"<span class='{badge_class}'>{role}</span>", unsafe_allow_html=True)
                    if role == "User":
                        prompt_text = msg.get("text", "")
                        if prompt_text:
                            st.markdown("**Prompt Text:**")
                            st.code(prompt_text)
                        prompt_evals = msg.get("prompt_evaluation", [])
                        if prompt_evals:
                            st.markdown("**Prompt Evaluations**")
                            eval_rows = []
                            for item in prompt_evals:
                                eval_rows.append({
                                    "Question": item.get("question", ""),
                                    "Description": item.get("description", ""),
                                    "Human Answer": item.get("human_input_value", ""),
                                })
                            st.table(pd.DataFrame(eval_rows))
                        other_fields = {k: v for k, v in msg.items() if k not in ["role", "text", "prompt_evaluation"]}
                        if other_fields:
                            st.markdown("**Additional Fields**")
                            for field, value in other_fields.items():
                                with st.expander(f"▸ {field}", expanded=False):
                                    if isinstance(value, (dict, list)):
                                        st.json(value)
                                    else:
                                        st.write(value)
                    elif role == "Assistant":
                        ideal_response = msg.get("signal", {}).get("ideal_response")
                        if ideal_response:
                            st.markdown("**💡 Ideal Response:**")
                            st.code(ideal_response)
                        human_evals = msg.get("signal", {}).get("human_evals", [])
                        for eval_set in human_evals:
                            eval_form = eval_set.get("evaluation_form", [])
                            if eval_form:
                                st.markdown("**🧾 Human Evaluation**")
                                eval_rows = []
                                for item in eval_form:
                                    eval_rows.append({
                                        "Question": item.get("question", ""),
                                        "Description": item.get("description", ""),
                                        "Human Answer": item.get("human_input_value", ""),
                                    })
                                st.table(pd.DataFrame(eval_rows))

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point"""
    st.title("🔍 Choose Viewer Mode")
    viewer_mode = st.radio(
        label="Select a Viewer",
        options=["Home", "RLHF Viewer", "JSON Visualizer", "Schema Validator", "JSON Mapper (Group & Aggregate)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    if viewer_mode == "RLHF Viewer":
        show_rlhf_viewer()
    elif viewer_mode == "JSON Visualizer":
        show_json_visualizer()
    elif viewer_mode == "Schema Validator":
        show_schema_validator()
    elif viewer_mode == "JSON Mapper (Group & Aggregate)":
        render_json_mapper_tab(prefer_global_csv=True)
    else:
        show_home()

def show_home():
    """Display home/welcome screen"""
    st.markdown("""
    ## Welcome to the RLHF & JSON Analysis Toolkit
    
    This application provides three main functions:
    
    1. **RLHF Viewer**: Inspect and analyze RLHF (Reinforcement Learning from Human Feedback) task data
    2. **JSON Visualizer**: Upload and analyze any JSON structure with AI-powered insights
    3. **Schema Validator**: Validate JSON documents against JSON schemas
    
    Select a mode from the tabs above to get started.
    """)
    st.image("https://via.placeholder.com/800x400?text=RLHF+Analysis+Tool", use_column_width=True)




def show_rlhf_viewer():
    """RLHF Viewer main interface"""
    st.title("🧠 RLHF Task Viewer")
    
    # Sidebar upload
    st.sidebar.header("📁 Upload RLHF Delivery JSON")
    json_file = st.sidebar.file_uploader(
        "Upload delivery JSON", 
        type="json", 
        key="delivery_json_uploader",
        help="Upload the RLHF delivery JSON file containing both rlhf and sft data"
    )
    
    if not json_file:
        st.info("👈 Upload a Delivery RLHF JSON to begin analysis")
        st.stop()
    
    # Load and validate data
    data = load_json(json_file)
    if not data or "rlhf" not in data or "sft" not in data:
        st.error("❌ Invalid RLHF JSON format. Must contain both 'rlhf' and 'sft' keys.")
        st.stop()
    
    # Process data
    rlhf_tasks = data["rlhf"]
    data_sft = data["sft"]
    
    annotator_task_df = RLHFProcessor.build_annotator_task_df(data_sft)
    task_id_map = {
        str(task["taskId"]): task 
        for task in rlhf_tasks 
        if "taskId" in task
    }
    
    if not task_id_map:
        st.warning("⚠️ No valid RLHF tasks with 'taskId' found")
        st.stop()
    
    data_csv = RLHFProcessor.compile_task_rows(task_id_map, data_sft, annotator_task_df)
    st.session_state["data_csv"] = data_csv
    
    # Display KPIs
    metrics = {
        "total_tasks": len(task_id_map),
        "unique_annotators": annotator_task_df["AnnotatorID"].nunique(),
        "unique_categories": data_csv["scope_category"].nunique() if "scope_category" in data_csv else 0,
        "unique_batches": data_csv["scope_batchId"].nunique() if "scope_batchId" in data_csv else 0
    }
    UIComponents.show_kpi_metrics(metrics)
    
    # Main tabs
    inspect_tab, csv_tab, delivery_tab, custom_tab = st.tabs([
        "🔍 Inspect Task ID",
        "🧾 View CSV", 
        "📦 Delivery Batch Creator",
        "🗂️ Custom Deliverable"
    ])
    
    with inspect_tab:
        st.subheader("Inspect a Single Task")
        selected_task_id = st.selectbox(
            "Select a Task ID", 
            list(task_id_map.keys()), 
            key="inspect_task_select"
        )
        
        if selected_task_id:
            task = task_id_map[selected_task_id]
            try:
                annotator_id = annotator_task_df[
                    annotator_task_df["task_id"] == int(selected_task_id)
                ]["AnnotatorID"].iloc[0]
            except (IndexError, ValueError):
                annotator_id = "NA"
            
            UIComponents.show_task_inspection(task, annotator_id)
    
    with csv_tab:
        st.subheader("Compiled Task Table")
        st.caption("Use ➕ Add filter to create filters across ANY column. If any advanced filter is active, pre-filters are ignored.")

        if data_csv.empty:
            st.warning("No compiled task data available")
        else:
            # 1) ALWAYS render the advanced filter builder so the ➕ button is visible
            st.markdown("### 🔎 Advanced filters")
            filtered_by_adv = UIComponents.filter_dataframe_advanced(data_csv)
            adv_active = UIComponents.adv_filters_active()

            # 2) If advanced filters are active → use them and skip pre-filters
            if adv_active:
                st.info("Advanced filters active → pre-filters are ignored.")
                final_df = filtered_by_adv
            else:
                # 3) Otherwise, optional pre-filters (your basic few-column ones)
                with st.expander("🔍 Optional pre-filters (basic, few columns)", expanded=False):
                    pre_filtered_df = UIComponents.filter_dataframe(data_csv)
                final_df = pre_filtered_df

            st.markdown(f"**Showing {len(final_df):,} of {len(data_csv):,} rows**")
            st.dataframe(final_df, use_container_width=True, height=480)
            
            st.markdown("---")
            _ = UIComponents.pivot_builder(final_df, title="📊 Build Pivot on Current Table")

            # ---------- Exports ----------
            st.markdown("<div class='rlhf-toolbar-sticky'></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                excel_name = st.text_input(
                    "Filename for full data export",
                    value="rlhf_tasks_full.xlsx",
                    key="excel_full_name"
                )
                st.download_button(
                    "⬇️ Download Full Data (Excel)",
                    data=RLHFProcessor.create_excel_export(data_csv),
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with col2:
                excel_filtered_name = st.text_input(
                    "Filename for filtered data export",
                    value="rlhf_tasks_filtered.xlsx",
                    key="excel_filtered_name"
                )
                st.download_button(
                    "⬇️ Download Filtered Data (Excel)",
                    data=RLHFProcessor.create_excel_export(final_df),
                    file_name=excel_filtered_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        

    with delivery_tab:
        st.subheader("📦 Delivery Batch Creator")
        st.markdown("""
        Create delivery batches from the compiled CSV data.  
        Upload the original ingestion JSON to match workitem IDs.
        """)
        
        ingestion_file = st.file_uploader(
            "Upload Ingestion JSON", 
            type="json", 
            key="delivery_ingestion_uploader"
        )
        
        if ingestion_file:
            ingestion_data = load_json(ingestion_file)
            if ingestion_data:
                pdf_col = st.selectbox(
                    "Select PDF name column for grouping",
                    options=data_csv.columns,
                    index=0,
                    key="delivery_pdf_col"
                )
                
                min_turns = st.number_input(
                    "Minimum turns per group",
                    min_value=1,
                    value=AppConfig.CUSTOM_DELIVERY_MIN_TURNS,
                    key="delivery_min_turns"
                )
                
                if st.button("🛠️ Build Delivery JSON"):
                    with st.spinner("Building delivery batch..."):
                        delivery_json = RLHFProcessor.build_delivery_json_from_ingestion(
                            ingestion_data,
                            data_csv,
                            data_sft,
                            pdf_col,
                            min_turns
                        )
                        
                        if delivery_json:
                            payload = json.dumps(delivery_json, indent=2).encode()
                            delivery_name = st.text_input(
                                "Filename for delivery export",
                                value=timestamped_json_filename(),
                                key="delivery_json_name"
                            )
                            st.download_button(
                                "⬇️ Download Delivery JSON",
                                data=payload,
                                file_name=delivery_name,
                                mime="application/json"
                            )
                            
                            st.markdown("#### 🔍 Preview of Delivery JSON")
                            st.json(delivery_json, expanded=False)
    
    # Custom Deliverable Builder
    CustomDeliverableBuilder.render_interface(custom_tab)

def show_json_visualizer():
    """JSON Visualizer interface"""
    st.title("📊 JSON Visualizer & Explainer")
    st.markdown("### 📁 Upload Your JSON File")
    
    json_file = st.file_uploader(
        "Upload JSON File", 
        type="json", 
        key="json_visualizer_uploader",
        help="Upload any JSON file for analysis and visualization"
    )
    
    if json_file:
        with st.spinner("Loading and analyzing JSON..."):
            json_data = load_json(json_file)
            
            if json_data:
                st.subheader("📄 Uploaded JSON Structure")
                with st.expander("View Raw JSON", expanded=False):
                    st.json(json_data)
                
                # AI-powered analysis
                if st.button("🧠 Analyze with AI", help="Get AI-powered insights about this JSON structure"):
                    load_dotenv()
                    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                    if not OPENAI_API_KEY:
                        st.error("❌ OpenAI API key not found in .env file")
                        st.stop()

                    client = openai.OpenAI(api_key=OPENAI_API_KEY)       
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
                    try:
                        response = client.chat.completions.create(
                            model=AppConfig.JSON_VISUALIZER_MODEL,
                            messages=[
                                {
                                    "role": "system", 
                                    "content": (
                                        "You are a reasoning expert and a JSON analysis specialist. "
                                        "Your job is to deeply analyze uploaded JSON data with step-by-step logic, "
                                        "clear formatting, and structured breakdowns."
                                    )
                                },
                                {
                                    "role": "user", 
                                    "content": prompt
                                }
                            ],
                            temperature=0.7,
                            max_tokens=AppConfig.JSON_VISUALIZER_MAX_TOKENS,
                            stream=True
                        )

                        full_output = ""
                        placeholder = st.empty()
                        
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                full_output += chunk.choices[0].delta.content
                                placeholder.markdown(full_output)
                        
                        st.markdown("### 🧠 Uploaded JSON: Full Breakdown and Cleaned Version")
                        st.markdown(full_output)

                    except Exception as e:
                        st.error(f"❌ Failed to analyze JSON due to: {str(e)}")

def show_schema_validator():
    """Schema Validator interface"""
    st.title("📐 Delivery-JSON Schema Validator")

    st.markdown("""
    Upload the final delivery batch JSON and a corresponding Draft-7 schema 
    to check structural compliance.
    """)

    col1, col2 = st.columns(2)
    with col1:
        delivery_file = st.file_uploader(
            "Upload delivery.json", 
            type="json", 
            key="validator_delivery"
        )
    with col2:
        schema_file = st.file_uploader(
            "Upload schema.json", 
            type="json", 
            key="validator_schema"
        )

    if delivery_file and schema_file:
        with st.spinner("Validating JSON against schema..."):
            delivery_data = load_json(delivery_file)
            schema = load_json(schema_file)
            
            if not delivery_data or not schema:
                st.error("❌ Failed to load one or both files")
                return
            
            try:
                errors = JSONValidator.validate_delivery(delivery_data, schema)
                
                if errors:
                    st.error(f"❌ Found {len(errors)} validation errors")
                    st.dataframe(pd.DataFrame(errors), use_container_width=True)
                else:
                    st.success("✅ Validation successful - all workitems conform to the schema!")
            
            except Exception as e:
                st.error(f"❌ Validation failed: {str(e)}")


if __name__ == "__main__":
    main()