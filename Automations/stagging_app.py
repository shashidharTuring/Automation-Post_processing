# Enhanced RLHF & JSON Viewer - Complete Professional Implementation
import streamlit as st
import json
import pandas as pd
import numpy as np
import io
import re
import uuid
from datetime import datetime, timezone
from jsonschema import Draft7Validator, ValidationError
from typing import Any, Dict, List, Optional, Tuple, Set, Union
import os
import copy
from dotenv import load_dotenv
import openai
import zipfile
from dataclasses import dataclass
from enum import Enum

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
# DATA MODELS
# =============================================================================
@dataclass
class SchemaOverride:
    """Data model for custom schema field overrides"""
    kind: str = "schema"
    col: str = ""
    pairs: List[Dict[str, str]] = None
    group_key: str = ""
    top_key: str = "items"

    def __post_init__(self):
        if self.pairs is None:
            self.pairs = []

class FieldKind(Enum):
    """Enum for schema field types in custom deliverable builder"""
    SCHEMA = "schema"
    STRING = "string"
    DICT = "dict"
    LIST_STRING = "list[string]"
    LIST_DICT = "list[dict]"
    DICT_LIST_DICT = "dict[list[dict]]"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_json(uploaded_file: Optional[io.BytesIO]) -> Optional[Union[dict, list]]:
    """Safely load JSON from uploaded file with error handling"""
    if uploaded_file is None:
        return None
    
    try:
        uploaded_file.seek(0)  # Rewind in case of multiple reads
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
    """
    Make a safe Excel/CSV column name while preserving original wording.
    Removes characters Excel dislikes and collapses multiple spaces.
    """
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

def extract_uuid_inner_keys(fmt_json: dict) -> list[str]:
    """
    Return EVERY raw key that occurs *inside* any UUID-named field under
    fmt_json["workitems"][0] (recursively, at any depth).
    """
    keys: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(k)
                walk(v)
        elif isinstance(obj, list) and obj:
            walk(obj[0])  # take structure of first element

    # find workitems[0] safely
    try:
        wi0 = fmt_json["workitems"][0]
    except Exception:
        return []

    # iterate UUID buckets only
    for k, v in wi0.items():
        if AppConfig.UUID_PATTERN.match(k):  # only process UUID-looking keys
            if isinstance(v, list) and v:
                walk(v[0])  # the list's first item shows the schema

    return sorted(keys)

def extract_leaf_paths(workitem: dict, sep: str = ".") -> list[str]:
    """
    Recursively return the dotted path for *every leaf value* inside workitem.
    Example path: "a2072a5f-3e48-... .0.data.taskAnswers.0.critique.details"
    """
    paths = []

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{sep}{k}" if prefix else k)
        elif isinstance(node, list) and node:
            walk(node[0], f"{prefix}{sep}0" if prefix else "0")
        else:
            paths.append(prefix)  # reached a scalar → leaf key

    walk(workitem)
    return sorted(paths)

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

                # User message processing
                if idx < len(users):
                    user_msg = users[idx]
                    row["prompt"] = user_msg.get("text", "")

                    # Prompt evaluations
                    for pe in user_msg.get("prompt_evaluation", []):
                        col = _clean_col(pe.get("question", ""))
                        row[col] = pe.get("human_input_value", "")
                    
                    # Additional user message fields
                    for k, v in user_msg.items():
                        if k not in {"role", "text", "prompt_evaluation"}:
                            row[f"user_{k}"] = _jsonify(v)
                else:
                    row["prompt"] = ""

                # Assistant message processing
                if idx < len(assistants):
                    assistant_msg = assistants[idx]
                    signal = assistant_msg.get("signal", {})
                    
                    # Model response
                    opts = assistant_msg.get("response_options", [])
                    row["model_response"] = opts[0].get("text", "") if opts else ""
                    
                    # Ideal response
                    row["ideal_response"] = signal.get("ideal_response", "")
                    
                    # Human evaluations
                    for he_set in signal.get("human_evals", []):
                        for item in he_set.get("evaluation_form", []):
                            col = _clean_col(item.get("question", ""))
                            row[col] = item.get("human_input_value", "")
                else:
                    row["model_response"] = ""
                    row["ideal_response"] = ""

                # Metadata counts
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
        answer = (
            row["ideal_response"]
            if "ideal_response" in row and not pd.isna(row["ideal_response"])
            else None
        )
        
        if "system_message" in eval_cols:
            system_message = (
                row["system_message"]
                if "system_message" in row and not pd.isna(row["system_message"])
                else None
            )
            return {
                "system_message": system_message,
                "question": question,
                "answer": answer,
                "metadata": meta
            }
        return {
            "question": question,
            "answer": answer,
            "metadata": meta
        }

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
        
        for index, work_id in enumerate(work_item_ids_needed):
            if index >= len(pdf_unique):
                break
                
            pdf_name = pdf_unique[index]
            data_tmp = data_csv.loc[data_csv[pdf_col] == pdf_name].reset_index(drop=True)
            
            if data_tmp.shape[0] < min_turns:
                skipped_groups.append((pdf_name, data_tmp.shape[0]))
                continue

            annot_vals = (
                data_tmp["AnnotatorID"].dropna().unique().tolist()
                if "AnnotatorID" in data_tmp
                else []
            )
            annotator_id = str(annot_vals[0]) if annot_vals else "NA"

            records = [
                RLHFProcessor.row_to_record(r, eval_cols) 
                for _, r in data_tmp.iterrows()
            ]
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
                errors.append(
                    JSONValidator._format_validation_error(validation_error, idx, workitem_id)
                )

        return errors

class CustomDeliverableBuilder:
    """Handles the custom deliverable builder functionality"""
    
    @staticmethod
    def _load_json(uploaded_file: Optional[io.BytesIO]) -> Optional[Union[dict, list]]:
        """Wrapper for consistent JSON loading"""
        return load_json(uploaded_file)

    @staticmethod
    def _default_for(jtype: str) -> Any:
        """Get default value for JSON type"""
        return {
            "object": {},
            "array": [],
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "string": ""
        }.get(jtype, "")

    @staticmethod
    def _coerce(val: Any, jtype: str) -> Any:
        """Coerce value to specified JSON type"""
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

    @staticmethod
    def gather_schema_leaves(
        node: Any, 
        path: str = "", 
        bag: Optional[Set[str]] = None, 
        tmap: Optional[Dict[str, str]] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """Recursively gather all leaf paths from JSON schema"""
        if bag is None:
            bag = set()
        if tmap is None:
            tmap = {}
            
        if isinstance(node, list):
            for sub in node:
                CustomDeliverableBuilder.gather_schema_leaves(
                    sub, f"{path}.0" if path else "0", bag, tmap
                )
            return sorted(bag), tmap
            
        if not isinstance(node, dict):
            bag.add(path)
            return sorted(bag), tmap
            
        for comb in ("oneOf", "anyOf", "allOf"):
            if comb in node:
                for sub in node[comb]:
                    CustomDeliverableBuilder.gather_schema_leaves(sub, path, bag, tmap)
                    
        if node.get("type") == "object" and "properties" in node:
            for k, v in node["properties"].items():
                CustomDeliverableBuilder.gather_schema_leaves(
                    v, f"{path}.{k}" if path else k, bag, tmap
                )
            return sorted(bag), tmap
            
        if node.get("type") == "array" and "items" in node:
            subs = node["items"] if isinstance(node["items"], list) else [node["items"]]
            for sub in subs:
                CustomDeliverableBuilder.gather_schema_leaves(
                    sub, f"{path}.0" if path else "0", bag, tmap
                )
            return sorted(bag), tmap
            
        bag.add(path)
        jtype = node.get("type", "string")
        tmap[path] = jtype if isinstance(jtype, str) else jtype[0]
        return sorted(bag), tmap

    @staticmethod
    def set_by_path(root: Any, path: str, value: Any) -> None:
        """Set value in nested structure using dotted path notation"""
        parts = path.split(".")
        current = root
        
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            is_index = part.isdigit()
            key = int(part) if is_index else part
            
            if is_last:
                if is_index:
                    while len(current) <= key:
                        current.append(None)
                    current[key] = value
                else:
                    current[key] = value
                return
                
            next_type = [] if parts[i + 1].isdigit() else {}
            
            if is_index:
                while len(current) <= key:
                    current.append(copy.deepcopy(next_type))
                if not isinstance(current[key], (list, dict)):
                    current[key] = copy.deepcopy(next_type)
                current = current[key]
            else:
                if key not in current or not isinstance(current[key], (list, dict)):
                    current[key] = copy.deepcopy(next_type)
                current = current[key]

    @staticmethod
    def _placeholder_for(col: str) -> str:
        """Generate placeholder text for column references"""
        return f"{{{{{col}}}}}" if col else ""

    @staticmethod
    def _type_default(jtype: Union[str, List[str]]) -> Any:
        """Get default value for JSON type, handling union types"""
        if isinstance(jtype, list):
            jtype = jtype[0] if jtype else "string"
        return CustomDeliverableBuilder._default_for(jtype)

    @staticmethod
    def _get_override(path: str) -> Dict[str, Any]:
        """Get override config for path from session state"""
        return st.session_state["ds_overrides"].get(path, {
            "kind": "schema",
            "col": "",
            "pairs": [],
            "group_key": "",
            "top_key": "items"
        })

    @staticmethod
    def _set_override(path: str, cfg: Dict[str, Any]) -> None:
        """Set override config for path in session state"""
        st.session_state["ds_overrides"][path] = cfg

    @staticmethod
    def _row_matches_group(df_row: Dict[str, Any], ref_row: Dict[str, Any], group_key: str) -> bool:
        """Check if two rows match on group key"""
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
        """Safely get value from row with empty string fallback"""
        v = row.get(col, "")
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else v

    @staticmethod
    def _materialize_struct_value(
        path: str,
        kind: str,
        cfg: Dict[str, Any],
        this_row: Dict[str, Any],
        all_df: pd.DataFrame,
        typemap: Dict[str, str]
    ) -> Any:
        """Generate concrete value for custom deliverable field"""
        if kind == "schema":
            col = cfg.get("col", "")
            if not col:
                return "__LEAVE_SKELETON__"
            jtype = typemap.get(path, "string")
            return CustomDeliverableBuilder._coerce(
                CustomDeliverableBuilder._val_from_col(this_row, col), 
                jtype
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

        if kind == "list[string]":
            gk, col = cfg.get("group_key", ""), cfg.get("col", "")
            if not gk or not col:
                return []
            ref = this_row
            items = []
            for _, r in all_df.iterrows():
                rd = r.to_dict()
                if CustomDeliverableBuilder._row_matches_group(rd, ref, gk):
                    v = CustomDeliverableBuilder._val_from_col(rd, col)
                    if v != "":
                        items.append(v)
            return items

        if kind == "list[dict]":
            gk, pairs = cfg.get("group_key", ""), cfg.get("pairs", [])
            if not gk or not pairs:
                return []
            ref = this_row
            items = []
            for _, r in all_df.iterrows():
                rd = r.to_dict()
                if CustomDeliverableBuilder._row_matches_group(rd, ref, gk):
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

        if kind == "dict[list[dict]]":
            top_key = (cfg.get("top_key") or "items").strip() or "items"
            gk, pairs = cfg.get("group_key", ""), cfg.get("pairs", [])
            if not gk or not pairs:
                return {top_key: []}
            ref = this_row
            items = []
            for _, r in all_df.iterrows():
                rd = r.to_dict()
                if CustomDeliverableBuilder._row_matches_group(rd, ref, gk):
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

        return "__LEAVE_SKELETON__"

    @staticmethod
    def render_interface(custom_tab) -> None:
        """Render the complete custom deliverable builder interface"""
        with custom_tab:
            st.header("🗂️ Custom Deliverable Builder")

            # Schema upload and data validation
            schema_file = st.file_uploader(
                "📄 Upload Output-Schema JSON", 
                type="json",
                help="Upload the JSON schema that defines your output structure"
            )
            schema_json = CustomDeliverableBuilder._load_json(schema_file)
            if schema_json is None:
                st.info("Upload a schema to start.")
                st.stop()

            data_csv = st.session_state.get("data_csv", pd.DataFrame())
            if data_csv.empty:
                st.error("CSV empty – load RLHF Viewer first.")
                st.stop()

            # Extract schema information
            leaves, typemap = CustomDeliverableBuilder.gather_schema_leaves(schema_json)

            # Initialize session state
            st.session_state.setdefault("use_custom_struct", False)
            st.session_state.setdefault("ds_overrides", {})
            st.session_state.setdefault("schema_mapping", {})

            # Toggle callback
            def toggle_callback():
                if st.session_state.use_custom_struct:
                    st.session_state.ds_overrides = {
                        p: {
                            "kind": "schema",
                            "col": "",
                            "pairs": [],
                            "group_key": "",
                            "top_key": "items"
                        }
                        for p in leaves
                    }

            # Master toggle
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
                    expand_all = st.checkbox(
                        "Expand all", 
                        value=True, 
                        key="custom_expand_all"
                    )
                with topc2:
                    ft = st.text_input(
                        "Filter fields (substring)", 
                        value="", 
                        key="custom_filter"
                    ).strip().lower()
                
                visible = [p for p in leaves if ft in p.lower()] if ft else leaves

                # Field type labels
                kind_labels = {
                    "schema": "Schema (typed default; or coerce from chosen column)",
                    "string": "string (raw from chosen column)",
                    "dict": "dict (key → column)",
                    "list[string]": "list[string] (group & column)",
                    "list[dict]": "list[dict] (group & pairs)",
                    "dict[list[dict]]": "dict[list[dict]] (top key, group & pairs)"
                }

                # Render each field configuration
                for p in visible:
                    ov = CustomDeliverableBuilder._get_override(p)
                    with st.expander(f"📝 {p}", expanded=expand_all):
                        current_kind = ov.get("kind", "schema")
                        kind_options = [
                            "schema", "string", "dict", 
                            "list[string]", "list[dict]", "dict[list[dict]]"
                        ]
                        kind_index = (
                            kind_options.index(current_kind) 
                            if current_kind in kind_options 
                            else 0
                        )
                        
                        kind = st.selectbox(
                            "Structure type",
                            options=kind_options,
                            format_func=lambda k: kind_labels[k],
                            index=kind_index,
                            key=f"kind_{_safe_key(p)}"
                        )
                        
                        if kind != current_kind:
                            ov = {
                                "kind": kind,
                                "col": "",
                                "pairs": [],
                                "group_key": "",
                                "top_key": "items"
                            }
                            CustomDeliverableBuilder._set_override(p, ov)

                        # Value columns for Schema & String
                        if kind in ("schema", "string"):
                            current_col = ov.get("col", "")
                            col_index = (
                                csv_cols.index(current_col) 
                                if current_col in csv_cols 
                                else 0
                            )
                            
                            ov["col"] = st.selectbox(
                                "Value column",
                                options=csv_cols,
                                index=col_index,
                                key=f"col_{_safe_key(p)}"
                            )
                            CustomDeliverableBuilder._set_override(p, ov)

                        # Grouping for list kinds
                        if kind in ("list[string]", "list[dict]", "dict[list[dict]]"):
                            c1, c2 = st.columns(2)
                            with c1:
                                current_group_key = ov.get("group_key", "")
                                group_index = (
                                    csv_cols.index(current_group_key) 
                                    if current_group_key in csv_cols 
                                    else 0
                                )
                                
                                ov["group_key"] = st.selectbox(
                                    "Group by column",
                                    options=csv_cols,
                                    index=group_index,
                                    key=f"group_key_{_safe_key(p)}"
                                )
                            with c2:
                                if kind == "list[string]":
                                    current_ls_col = ov.get("col", "")
                                    ls_col_index = (
                                        csv_cols.index(current_ls_col) 
                                        if current_ls_col in csv_cols 
                                        else 0
                                    )
                                    
                                    ov["col"] = st.selectbox(
                                        "Element column (string)",
                                        options=csv_cols,
                                        index=ls_col_index,
                                        key=f"ls_col_{_safe_key(p)}"
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
                                    "col": st.column_config.SelectboxColumn(
                                        "CSV Column", 
                                        options=csv_cols
                                    )
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
                tbl_df = pd.DataFrame({
                    "key": leaves,
                    "csv_column": [mapping.get(k, "") for k in leaves]
                })

                Selectbox = getattr(st.column_config, "SelectboxColumn", None)
                col_cfg = {
                    "key": st.column_config.Column(disabled=True),
                    "csv_column": (
                        Selectbox("Map to CSV (blank = unmapped)", options=csv_cols)
                        if Selectbox 
                        else st.column_config.Column("Map to CSV", width="medium")
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

            # Skeleton template preview
            st.markdown("#### 📄 Skeleton Template")
            skeleton: Dict[str, Any] = {}

            if use_custom:
                for p in leaves:
                    ov = CustomDeliverableBuilder._get_override(p)
                    kind = ov.get("kind", "schema")
                    
                    if kind == "schema":
                        col = ov.get("col", "")
                        CustomDeliverableBuilder.set_by_path(
                            skeleton, 
                            p,
                            CustomDeliverableBuilder._placeholder_for(col) if col 
                            else CustomDeliverableBuilder._type_default(typemap.get(p, "string")))
                    
                    elif kind == "string":
                        col = ov.get("col", "")
                        CustomDeliverableBuilder.set_by_path(
                            skeleton,
                            p,
                            CustomDeliverableBuilder._placeholder_for(col) if col 
                            else ""
                        )
                    
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
                        CustomDeliverableBuilder.set_by_path(
                            skeleton,
                            p,
                            [
                                CustomDeliverableBuilder._placeholder_for(col) if col else "",
                                "..."
                            ]
                        )
                    
                    elif kind == "list[dict]":
                        ex = {}
                        for pair in ov.get("pairs", []):
                            k = (pair.get("key") or "").strip()
                            col = pair.get("col", "")
                            if k:
                                ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
                        CustomDeliverableBuilder.set_by_path(
                            skeleton,
                            p,
                            [ex, "..."]
                        )
                    
                    elif kind == "dict[list[dict]]":
                        top_key = (ov.get("top_key") or "items").strip() or "items"
                        ex = {}
                        for pair in ov.get("pairs", []):
                            k = (pair.get("key") or "").strip()
                            col = pair.get("col", "")
                            if k:
                                ex[k] = CustomDeliverableBuilder._placeholder_for(col) if col else ""
                        CustomDeliverableBuilder.set_by_path(
                            skeleton,
                            p,
                            {top_key: [ex, "..."]}
                        )
                    
                    else:
                        CustomDeliverableBuilder.set_by_path(
                            skeleton,
                            p,
                            CustomDeliverableBuilder._type_default(typemap.get(p, "string")))
            else:
                for p in leaves:
                    jtype = typemap.get(p, "string")
                    col = mapping.get(p, "")
                    CustomDeliverableBuilder.set_by_path(
                        skeleton,
                        p,
                        CustomDeliverableBuilder._placeholder_for(col) if col 
                        else CustomDeliverableBuilder._type_default(jtype)
                    )

            st.json(skeleton, expanded=False)

            # Generate deliverables
            if st.button("⚙️ Generate deliverables.json"):
                filled_list = []
                
                if use_custom:
                    for _, row in data_csv.iterrows():
                        this = copy.deepcopy(skeleton)
                        rd = row.to_dict()
                        for p in leaves:
                            ov = CustomDeliverableBuilder._get_override(p)
                            kind = ov.get("kind", "schema")
                            val = CustomDeliverableBuilder._materialize_struct_value(
                                p, kind, ov, rd, data_csv, typemap
                            )
                            if val != "__LEAVE_SKELETON__":
                                CustomDeliverableBuilder.set_by_path(this, p, val)
                        filled_list.append(this)
                else:
                    for _, row in data_csv.iterrows():
                        this = copy.deepcopy(skeleton)
                        rd = row.to_dict()
                        for p in leaves:
                            col = mapping.get(p, "")
                            jtype = typemap.get(p, "string")
                            raw = rd.get(col, "") if col else ""
                            val = CustomDeliverableBuilder._coerce(raw, jtype)
                            CustomDeliverableBuilder.set_by_path(this, p, val)
                        filled_list.append(this)

                if filled_list:
                    payload = json.dumps(filled_list, ensure_ascii=False, indent=2).encode()
                    custom_name = st.text_input(
                        "Filename for custom deliverables",
                        value="deliverables.json",
                        key="custom_deliverables_name"
                    )
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

        # 2-column grid for filter widgets
        n_cols = 2
        filter_cols = st.columns(n_cols)
        selected_filters = []

        for i, col in enumerate(cols_present):
            idx = i % n_cols
            with filter_cols[idx]:
                series = df[col].dropna()
                unique_values = series.unique().tolist()
                
                # Create mapping for display labels to actual values
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

        # Display active filters as chips
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
            
            # Colab link
            colab_link = task_data.get("task", {}).get("colabLink", "N/A")
            if colab_link != "N/A":
                st.markdown(f"🔗 **RLHF Link**: [Open Task]({colab_link})")
            else:
                st.markdown("🔗 **RLHF Link**: Not available")
            
            # Scope requirements
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
                        # User message content
                        prompt_text = msg.get("text", "")
                        if prompt_text:
                            st.markdown("**Prompt Text:**")
                            st.code(prompt_text)
                        
                        # Prompt evaluations
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
                        
                        # Additional fields
                        other_fields = {
                            k: v for k, v in msg.items() 
                            if k not in ["role", "text", "prompt_evaluation"]
                        }
                        if other_fields:
                            st.markdown("**Additional Fields**")
                            for field, value in other_fields.items():
                                with st.expander(f"▸ {field}", expanded=False):
                                    if isinstance(value, (dict, list)):
                                        st.json(value)
                                    else:
                                        st.write(value)
                    
                    elif role == "Assistant":
                        # Ideal response
                        ideal_response = msg.get("signal", {}).get("ideal_response")
                        if ideal_response:
                            st.markdown("**💡 Ideal Response:**")
                            st.code(ideal_response)
                        
                        # Human evaluations
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
    
    # Navigation tabs
    viewer_mode = st.radio(
        label="Select a Viewer",
        options=["Home", "RLHF Viewer", "JSON Visualizer", "Schema Validator"],
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
        st.caption("Filters available for: task_id, scope_category, scope_batchName, scope_batchId")
        
        if data_csv.empty:
            st.warning("No compiled task data available")
        else:
            with st.expander("🔍 Add Filters", expanded=False):
                filtered_df = UIComponents.filter_dataframe(data_csv)
            
            st.markdown(
                f"**Showing {len(filtered_df):,} of {len(data_csv):,} rows**",
                unsafe_allow_html=True
            )
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Excel export
            st.markdown("<div class='rlhf-toolbar-sticky'></div>", unsafe_allow_html=True)
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                excel_name = st.text_input(
                    "Filename for full data export", 
                    value="rlhf_tasks_full.xlsx",
                    key="excel_full_name"
                )
                st.download_button(
                    label="⬇️ Download Full Data (Excel)",
                    data=RLHFProcessor.create_excel_export(data_csv),
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with export_col2:
                excel_filtered_name = st.text_input(
                    "Filename for filtered data export", 
                    value="rlhf_tasks_filtered.xlsx",
                    key="excel_filtered_name"
                )
                st.download_button(
                    label="⬇️ Download Filtered Data (Excel)",
                    data=RLHFProcessor.create_excel_export(filtered_df),
                    file_name=excel_filtered_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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