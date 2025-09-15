# json_mapper_tab.py
# ------------------------------------------------------------
# Drop-in tab that embeds the full "Code 1" JSON Mapper (Group,
# Aggregate, Dictionary Lists) into a function you can call
# from Code 2 without modifying any existing behavior.
#
# - No set_page_config here (avoids clashes).
# - All helpers are namespaced jm_* to avoid conflicts.
# - Uses st.session_state["jm_*"] keys.
# - HTML/JS block is a raw string + .replace(...) to avoid
#   Python parsing JS "try { ... }".
# - Fixes the '✎ static edit' handler rule assignment: rules[p].
# ------------------------------------------------------------

import io, json, copy
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================== Utils ===========================
def jm_load_json(uploaded_file: Optional[io.BytesIO]) -> Optional[Union[dict, list]]:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        return json.load(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to load JSON: {e}")
        return None

def jm_read_table(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    name = (uploaded.name or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Failed to read file '{uploaded.name}': {e}")
        return None

def _jm_base_type(t):
    if isinstance(t, list):
        return t[0] if t else "string"
    return t or "string"

def jm_target_type_for(path: str, typemap: Dict[str, Union[str, List[str]]], overrides: Dict[str, str]) -> str:
    if overrides and path in overrides and overrides[path]:
        return overrides[path]
    return _jm_base_type(typemap.get(path, "string"))

def jm_type_default(jtype: Union[str, List[str]]) -> Any:
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
        for t in ("object", "array"):
            if t in jtype: return _d(t)
        for t in ("string", "number", "integer", "boolean"):
            if t in jtype: return _d(t)
        return _d(jtype[0]) if jtype else ""
    return _d(jtype)

def jm_set_by_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    parent = None
    parent_key = None
    parent_is_index = False

    def _install_in_parent(new_container):
        nonlocal current
        if parent is None:
            if isinstance(root, dict):
                root.clear()
                if isinstance(new_container, dict):
                    root.update(new_container); current = root
                else:
                    current = new_container
            else:
                current = new_container
        else:
            if parent_is_index:
                while len(parent) <= parent_key: parent.append(None)
                parent[parent_key] = new_container
            else:
                parent[parent_key] = new_container
            current = new_container

    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        is_index = part.isdigit()
        key = int(part) if is_index else part
        next_container = [] if (not is_last and parts[i+1].isdigit()) else {}

        if is_index:
            if not isinstance(current, list): _install_in_parent([])
            while len(current) <= key: current.append(copy.deepcopy(next_container))
            if is_last:
                current[key] = value; return
            if not isinstance(current[key], (list, dict)):
                current[key] = copy.deepcopy(next_container)
            parent, parent_key, parent_is_index = current, key, True
            current = current[key]
        else:
            if not isinstance(current, dict): _install_in_parent({})
            if is_last:
                current[key] = value; return
            if key not in current or not isinstance(current[key], (list, dict)):
                current[key] = copy.deepcopy(next_container)
            parent, parent_key, parent_is_index = current, key, False
            current = current[key]

def jm_delete_by_path(root: Any, path: str) -> None:
    parts = path.split(".")
    current = root
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        is_index = part.isdigit()
        if is_index:
            idx = int(part)
            if not isinstance(current, list) or idx >= len(current): return
            if is_last:
                current[idx] = None
                return
            current = current[idx]
        else:
            if not isinstance(current, dict) or part not in current: return
            if is_last:
                del current[part]
                return
            current = current[part]

def jm_coerce(val: Any, jtype: str) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return jm_type_default(jtype)
    try:
        if jtype == "string":
            return str(val)
        if jtype == "integer":
            return int(float(val))
        if jtype == "number":
            return float(val)
        if jtype == "boolean":
            if isinstance(val, bool): return val
            return str(val).strip().lower() in ("true","1","yes","y")
        if jtype == "array":
            if isinstance(val, (list, dict)): return val
            return [v.strip() for v in str(val).split(",") if v.strip()]
        if jtype == "object":
            if isinstance(val, (dict, list)): return val
            try: return json.loads(val)
            except: return {}
    except:
        return jm_type_default(jtype)
    return val

def jm_parse_static(value_str: str, static_type: str) -> Any:
    if static_type == "null": return None
    if static_type == "string": return "" if value_str is None else str(value_str)
    if static_type == "integer":
        try: return int(float(value_str))
        except: return 0
    if static_type == "number":
        try: return float(value_str)
        except: return 0.0
    if static_type == "boolean":
        return str(value_str).strip().lower() in ("true","1","yes","y")
    if static_type in ("object","array"):
        try: return json.loads(value_str or ("{}" if static_type=="object" else "[]"))
        except: return {} if static_type=="object" else []
    return value_str

# =========================== Schema walk ===========================
def jm_canonicalize_schema_node(node: Any) -> str:
    DROP = {"title", "description", "examples"}
    def strip_meta(obj):
        if isinstance(obj, dict):
            return {k: strip_meta(v) for k, v in obj.items() if k not in DROP}
        if isinstance(obj, list):
            return [strip_meta(x) for x in obj]
        return obj
    try: return json.dumps(strip_meta(node), sort_keys=True)
    except: return str(type(node))

def jm_gather_schema_paths(
    node: Any, path: str = "", seen: Optional[set] = None, bag: Optional[set] = None,
    tmap: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> Tuple[List[str], Dict[str, Union[str, List[str]]]]:
    if seen is None: seen = set()
    if bag is None: bag = set()
    if tmap is None: tmap = {}

    if isinstance(node, dict):
        nid = id(node)
        if nid in seen: return sorted(bag), tmap
        seen.add(nid)

    def _merge_type(p: str, t: Optional[Union[str, List[str]]]):
        if not p: return
        types = ["string"] if t is None else ([str(x) for x in t] if isinstance(t, list) else [str(t)])
        if p not in tmap:
            tmap[p] = types[0] if len(types)==1 else sorted(set(types))
        else:
            prev = tmap[p] if isinstance(tmap[p], list) else [tmap[p]]
            merged = sorted(set(prev).union(types))
            tmap[p] = merged[0] if len(merged)==1 else merged
        bag.add(p)

    if isinstance(node, dict):
        declared_type = node.get("type")
        if declared_type in ("object","array","string","number","integer","boolean"):
            _merge_type(path, declared_type)
        elif "properties" in node or "items" in node:
            _merge_type(path, "object" if "properties" in node else "array")

        props = node.get("properties")
        if isinstance(props, dict) and props:
            for k, v in props.items():
                new_path = f"{path}.{k}" if path else k
                jm_gather_schema_paths(v, new_path, seen, bag, tmap)

        if "items" in node:
            items = node["items"]
            if isinstance(items, list) and items:
                sig_to_first_idx: Dict[str, int] = {}
                for i, sub in enumerate(items):
                    sig = jm_canonicalize_schema_node(sub)
                    if sig in sig_to_first_idx: continue
                    sig_to_first_idx[sig] = i
                    idx_path = f"{path}.{i}" if path else str(i)
                    jm_gather_schema_paths(sub, idx_path, seen, bag, tmap)
            else:
                idx_path = f"{path}.0" if path else "0"
                jm_gather_schema_paths(items, idx_path, seen, bag, tmap)

        for comb in ("oneOf","anyOf","allOf"):
            opts = node.get(comb)
            if isinstance(opts, list) and opts:
                for sub in opts:
                    jm_gather_schema_paths(sub, path, seen, bag, tmap)

        return sorted(bag), tmap

    if isinstance(node, list):
        for i, sub in enumerate(node):
            idx_path = f"{path}.{i}" if path else str(i)
            jm_gather_schema_paths(sub, idx_path, seen, bag, tmap)
        return sorted(bag), tmap

    return sorted(bag), tmap

def jm_prune_array_item_exemplars(
    leaves: List[str], typemap: Dict[str, Union[str, List[str]]],
    *, drop_scalar_items: bool = True, drop_object_item_containers: bool = True,
) -> Tuple[List[str], Dict[str, Union[str, List[str]]]]:
    leafset = set(leaves)

    if drop_scalar_items:
        to_remove = []
        for p in list(leafset):
            if not p.endswith(".0"): continue
            parent = p[:-2]
            parent_t = _jm_base_type(typemap.get(parent, ""))
            item_t = _jm_base_type(typemap.get(p, "string"))
            if parent in leafset and parent_t == "array" and item_t in ("string","number","integer","boolean"):
                to_remove.append(p)
        for p in to_remove:
            leafset.discard(p); typemap.pop(p, None)

    if drop_object_item_containers:
        to_remove = []
        for p in list(leafset):
            if not p.endswith(".0"): continue
            has_children = any(x.startswith(p + ".") for x in leafset)
            if has_children and _jm_base_type(typemap.get(p, "object")) == "object":
                to_remove.append(p)
        for p in to_remove:
            leafset.discard(p); typemap.pop(p, None)

    return sorted(leafset), typemap

def jm_all_branch_paths_from_leaves(leaves: List[str]) -> List[str]:
    out = set()
    for p in leaves:
        parts = p.split(".")
        for i in range(1, len(parts)+1):
            out.add(".".join(parts[:i]))
    return sorted(out, key=lambda s: (s.count("."), s))

# =========================== DnD Panel (Columns + Static + Dict + Agg) ===========================
def jm_render_dnd_panel(
    csv_cols: List[str],
    skeleton: dict,
    drop_paths: List[str],
    typemap: Dict[str, Union[str, List[str]]],
    arrays_scalar: set,
    existing_rules: Dict[str, dict],
    existing_hidden: List[str],
    existing_excluded: List[str],
    all_branches: List[str],
    existing_types: Dict[str, str],
    existing_objrules: Dict[str, list],
    existing_aggrules: Dict[str, str],
):
    import hashlib

    token_map = {}
    tok_skel = copy.deepcopy(skeleton)

    def base_type(t):
        if isinstance(t, list): return t[0] if t else "string"
        return t or "string"

    for path in drop_paths:
        tok = "__DROP__::" + hashlib.md5(path.encode()).hexdigest()
        token_map[path] = tok
        is_array_parent = (base_type(typemap.get(path, "")) == "array")
        if is_array_parent and path in arrays_scalar:
            jm_set_by_path(tok_skel, path, [tok])
        else:
            jm_set_by_path(tok_skel, path, tok)

    JS_COLS      = json.dumps(csv_cols or [], ensure_ascii=False)
    JS_TOKENIZED = json.dumps(tok_skel, ensure_ascii=False)
    JS_TOKEN_MAP = json.dumps(token_map, ensure_ascii=False)
    JS_RULES     = json.dumps(existing_rules or {}, ensure_ascii=False)
    JS_HIDDEN    = json.dumps(list(existing_hidden or []), ensure_ascii=False)
    JS_EXCLUDED  = json.dumps(list(existing_excluded or []), ensure_ascii=False)
    JS_BRANCHES  = json.dumps(all_branches or [], ensure_ascii=False)
    JS_TYPES     = json.dumps(existing_types or {}, ensure_ascii=False)
    JS_OBJRULES  = json.dumps(existing_objrules or {}, ensure_ascii=False)
    JS_AGGRULES  = json.dumps(existing_aggrules or {}, ensure_ascii=False)

    # Raw HTML (no f-strings) + .replace injections to keep JS intact.
    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:0; padding:10px; }
  .wrap { display:grid; grid-template-columns: 1fr 2fr; gap:16px; }
  .panel { border:1px solid #e1e4ea; border-radius:8px; padding:12px; background:#f9fafc; }
  .title { margin:0 0 8px 0; font-weight:600; }
  .column-list { max-height: 420px; overflow:auto; background:#fff; border:1px solid #eee; border-radius:6px; padding:8px; }
  .drag-item { padding:8px 10px; margin:6px 0; background:#fff; border:1px solid #d8dee8; border-radius:6px; cursor:grab; user-select:none; }
  .chip { display:inline-block; padding:6px 10px; margin:6px 0 10px 0; background:#fff; border:1px dashed #8b9dcf; border-radius:16px; cursor:grab; user-select:none; }
  .chip span { font-weight:600; color:#435; }

  .jsonbox { max-height: 700px; overflow:auto; background:#fff; border:1px solid #eee; border-radius:6px; padding:12px; }
  pre { margin:0; white-space: pre-wrap; word-break: break-word; }
  .dz {
    display:inline-flex; align-items:center; gap:6px;
    min-width: 200px; padding:2px 6px; margin:0 2px;
    border:2px dashed #cfd7e6; border-radius:6px; background:#fcfdff;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .dz.active { background:#e8f1ff; border-color:#5a8dee; }
  .label-empty { color:#98a0ad; }
  .label-bound { color:#1e88e5; font-weight:600; }
  .label-type  { color:#6b21a8; font-weight:600; }
  .label-agg   { color:#0f766e; font-weight:700; }

  .btnx, .btns, .btnclr, .btntype {
    border:0; border-radius:6px; cursor:pointer; padding:0 6px; font-weight:700; height:22px;
  }
  .btnx { background:#ffe6e8; color:#b00020; }
  .btns { background:#e9f7ef; color:#1b5e20; }
  .btnclr { background:#f3f3f3; color:#333; }
  .btntype { background:#ede9fe; color:#4c1d95; }

  .hint { color:#687388; font-size:12px; margin-top:4px; }

  .mgr { background:#fff; border:1px solid #e1e4ea; border-radius:8px; padding:8px; margin-top:10px; }
  .mgr input, .mgr select, .mgr button { height:28px; }
  .mgr .row { display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; }
  .mgr .row > * { flex: 1 0 160px; }

  textarea { width:100%; height:120px; border:1px solid #ddd; border-radius:6px; padding:8px; font-family: ui-monospace, monospace; }
  .toolbar { display:flex; gap:8px; align-items:center; margin-top:8px; flex-wrap:wrap; }
  .btn { border:1px solid #d8dee8; background:#fff; border-radius:6px; padding:6px 10px; cursor:pointer; }
  .btn.primary { border-color:#cdd8ff; color:#2643e9; }

  .removed { color:#c0c4cc; font-style:italic; }

  #typePanel { display:none; position:fixed; right:24px; top:24px; width:740px; max-width:calc(100vw - 48px);
               max-height:90vh; overflow:auto; background:#fff; border:1px solid #e1e4ea; border-radius:14px;
               box-shadow:0 18px 48px rgba(0,0,0,0.18); padding:16px; z-index:2147483647; }
  #typePanel h4 { margin:6px 0 10px; font-size:16px; }
  #typePanel .row { display:flex; gap:10px; margin:8px 0; flex-wrap:wrap; }
  #typePanel .row > * { flex: 1 0 220px; }
  #typePanel select, #typePanel input { height:34px; border-radius:8px; width:100%; }
  #dictBox { display:none; border:1px dashed #d8dee8; border-radius:10px; padding:10px; background:#fbfcff; }
  .dict-head { display:grid; grid-template-columns: 1.5fr 1fr 1.6fr 1fr 1.6fr 1fr 60px; gap:8px; font-size:12px; color:#445; }
  .dict-row  { display:grid; grid-template-columns: 1.5fr 1fr 1.6fr 1fr 1.6fr 1fr 60px; gap:8px; margin:6px 0; }
  .dict-row input, .dict-row select { height:32px; }
  .delbtn { height:32px; border:0; border-radius:6px; background:#ffe6e8; color:#912; cursor:pointer; }
</style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h3 class="title">📊 CSV/Excel Columns (drag)</h3>
      <div id="cols" class="column-list"></div>
      <div class="chip" draggable="true" id="staticChip"><span>Static value</span></div>
      <div class="hint">Drag a column or the static chip onto a placeholder in the JSON.</div>

      <div class="mgr">
        <h4 style="margin:4px 0;">🧩 Branch Manager (Hide/Exclude)</h4>
        <div class="row">
          <input id="filter" placeholder="Filter paths..."/>
          <select id="paths" size="6" multiple></select>
        </div>
        <div class="row">
          <button id="hide"   title="Set selected paths to null">Hide (null)</button>
          <button id="unhide" title="Remove from hidden">Unhide</button>
        </div>
        <div class="row">
          <button id="exclude"   title="Remove selected paths">Exclude (remove)</button>
          <button id="unexclude" title="Remove from excluded">Un-exclude</button>
        </div>
      </div>

      <div class="toolbar">
        <button id="apply" class="btn primary">Apply mapping</button>
        <button id="clearAll" class="btn">Clear mappings</button>
        <button id="restoreAll" class="btn">Restore hidden</button>
        <button id="copy" class="btn">Copy State JSON</button>
      </div>
      <textarea id="stateOut" readonly></textarea>
    </section>

    <section class="panel">
      <h3 class="title">🗂️ JSON Skeleton (drop onto placeholders)</h3>
      <div class="hint">✎ edit static • 🗑 clear mapping • × hide (null) • ⚙ set type / dictionary / aggregation</div>
      <div id="jsonbox" class="jsonbox"><pre id="jsonpre"></pre></div>
    </section>
  </div>

  <!-- Floating type + dictionary + aggregation panel -->
  <div id="typePanel" aria-hidden="true">
    <h4>⚙ Field Settings</h4>
    <div class="row">
      <input id="tp_path" readonly />
      <select id="tp_type" title="Datatype">
        <option value="">(inherit schema)</option>
        <option value="string">string</option>
        <option value="number">number</option>
        <option value="integer">integer</option>
        <option value="boolean">boolean</option>
        <option value="null">null</option>
        <option value="array">array</option>
        <option value="object">object</option>
        <option value="dictionary">dictionary</option>
      </select>
      <select id="tp_agg" title="Aggregation (when grouped)">
        <option value="">(no aggregation)</option>
        <option value="unique">unique</option>
        <option value="list">list</option>
        <option value="dictlist">list of dictionaries</option>
      </select>
    </div>

    <div id="dictBox">
      <h4 style="margin:4px 0 6px;">Dictionary builder</h4>
      <div class="dict-head">
        <div>Key</div><div>Source</div><div>CSV Column</div><div>Static Type</div><div>Static Value</div><div>As Type</div><div></div>
      </div>
      <div id="dictRows"></div>
      <div class="row" style="justify-content:flex-start;">
        <button id="addRow" class="btn">➕ Add entry</button>
      </div>
      <div class="hint">“As Type” casts each value to string/number/integer/boolean/null/object/array.</div>
    </div>

    <div class="row" style="justify-content:flex-end; margin-top:10px;">
      <button id="tp_close" class="btn">Close</button>
      <button id="tp_save" class="btn primary">Save</button>
    </div>
  </div>

<script>
  const CSV_COLS  = __COLS__;
  const TOKENIZED = __TOKENIZED__;
  const TOKEN_MAP = __TOKEN_MAP__;
  let rules    = __RULES__;
  let hidden   = __HIDDEN__;
  let excluded = __EXCLUDED__;
  let types    = __TYPES__;     // path -> "string|number|integer|boolean|null|object|array"
  let objrules = __OBJRULES__;  // path -> [{key, mode, csv_col, static_type, static_value, as_type}]
  let aggrules = __AGGRULES__;  // path -> "unique|list|dictlist"
  const ALL_PATHS = __BRANCHES__;
  let CURRENT_TYPE_PATH = "";
  let PANEL_ROWS = [];

  function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
  function reEsc(s) { return String(s).replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&"); }
  function isHiddenPath(p) { return hidden.includes(p); }

  function hidePath(p) {
    if (!hidden.includes(p)) hidden.push(p);
    Object.keys(rules).forEach(k => { if (k===p || k.startsWith(p+'.')) delete rules[k]; });
    render(); push(true);
  }
  function unhidePaths(ps) { hidden = hidden.filter(x => !ps.includes(x)); render(); push(true); }
  function excludePaths(ps) {
    ps.forEach(p => { if (!excluded.includes(p)) excluded.push(p); });
    Object.keys(rules).forEach(k => { if (ps.some(p => k===p || k.startsWith(p+'.'))) delete rules[k]; });
    render(); push(true);
  }
  function unexcludePaths(ps) { excluded = excluded.filter(x => !ps.includes(x)); render(); push(true); }

  function renderCols() {
    const root = document.getElementById("cols");
    root.innerHTML = "";
    (CSV_COLS || []).filter(Boolean).forEach(col => {
      const d = document.createElement("div");
      d.className = "drag-item";
      d.textContent = col;
      d.draggable = true;
      d.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", "CSV::"+col); });
      root.appendChild(d);
    });
  }

  function tryParseStatic(val, t) {
    try {
      if (t === "null") return null;
      if (t === "string") return String(val);
      if (t === "integer") return parseInt(val);
      if (t === "number") return parseFloat(val);
      if (t === "boolean") return ["true","1","yes","y"].includes(String(val).toLowerCase());
      if (t === "object" || t === "array") return JSON.parse(val || (t==="object"?"{}":"[]"));
    } catch(e) {}
    return val;
  }

  function push(force=false) {
    const out = { rules, hidden, excluded, types, objrules, aggrules };
    if (force) out._ping = (out._ping || 0) + 1;
    const ta = document.getElementById("stateOut");
    if (ta) ta.value = JSON.stringify(out, null, 2);
    if (window && window.Streamlit && typeof window.Streamlit.setComponentValue === "function") {
      try { window.Streamlit.setComponentValue(out); } catch (e) {}
    }
  }

  function renderPathsList() {
    const filter = (document.getElementById("filter").value || "").toLowerCase();
    const sel = document.getElementById("paths");
    sel.innerHTML = "";
    ALL_PATHS.filter(p => p.toLowerCase().includes(filter)).forEach(p => {
      const dictBadge = (objrules[p] && objrules[p].length) ? ("  [dict:"+objrules[p].length+" keys]") : "";
      const aggBadge = aggrules[p] ? ("  [agg:"+aggrules[p]+"]") : "";
      const typeBadge = types[p] ? ("  [as:" + types[p] + "]") : "";
      const hiddenBadge = hidden.includes(p) ? "  [hidden]" : "";
      const excludedBadge = excluded.includes(p) ? "  [excluded]" : "";
      const opt = document.createElement("option");
      opt.value = p; opt.textContent = p + hiddenBadge + excludedBadge + typeBadge + dictBadge + aggBadge;
      sel.appendChild(opt);
    });
  }

  function render() {
    const raw = JSON.stringify(TOKENIZED, null, 2);
    let html = esc(raw);

    for (const [path, tok] of Object.entries(TOKEN_MAP)) {
      const needleQuoted = '\\"' + tok + '\\"';
      const re = new RegExp(reEsc(needleQuoted), "g");

      if (excluded.some(e => path === e || path.startsWith(e+'.'))) {
        html = html.replace(re, '<span class="removed">/* excluded */</span>');
        continue;
      }
      if (isHiddenPath(path)) {
        html = html.replace(re, '<span class="removed">null</span>');
        continue;
      }

      const r = rules[path];
      let boundText = "";
      if (r && r.mode === "csv") boundText = "← " + esc(r.csv_col);
      if (r && r.mode === "static") boundText = "=" + esc(JSON.stringify(r.static_value));
      const ty = types[path] ? ' <span class="label-type">(as:' + esc(types[path]) + ')</span>' : '';
      const ob = (objrules[path] && objrules[path].length) ? ' <span class="label-type">[dict:'+objrules[path].length+' keys]</span>' : '';
      const ag = aggrules[path] ? ' <span class="label-agg">[agg:'+esc(aggrules[path])+']</span>' : '';

      const ph = boundText
        ? '<span class="label-bound">' + boundText + '</span>' + ty + ob + ag
        : '<span class="label-empty">Drop column/static here</span>' + ty + ob + ag;

      const zone =
        '<span class="dz" data-path="' + path + '">' +
          ph +
          '<button class="btntype" title="Set type / dictionary / aggregation" data-type="' + path + '">⚙</button>' +
          '<button class="btns" title="Edit static" data-edit="' + path + '">✎</button>' +
          '<button class="btnclr" title="Clear mapping" data-clear="' + path + '">🗑</button>' +
          '<button class="btnx" title="Hide this key (null)" data-remove="' + path + '">×</button>' +
        '</span>';

      html = html.replace(re, zone);
    }

    const pre = document.getElementById("jsonpre");
    pre.innerHTML = html;

    const chip = document.getElementById("staticChip");
    if (chip) chip.addEventListener("dragstart", e => e.dataTransfer.setData("text/plain", "STATIC::"));

    document.querySelectorAll(".dz").forEach(zone => {
      zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("active"); });
      zone.addEventListener("dragleave", () => zone.classList.remove("active"));
      zone.addEventListener("drop", e => {
        e.preventDefault();
        zone.classList.remove("active");
        const data = e.dataTransfer.getData("text/plain");
        const path = zone.getAttribute("data-path");
        if (!path) return;

        if (data.startsWith("CSV::")) {
          const col = data.substring(5);
          rules[path] = { mode:"csv", csv_col: col };
          render(); push(true);
        } else if (data.startsWith("STATIC::")) {
          const sval = prompt("Enter static value (string for most; JSON for object/array):", "");
          if (sval === null) return;
          let stype = types[path] || "string";
          rules[path] = { mode:"static", static_type: stype, static_value: tryParseStatic(sval, stype) };
          render(); push(true);
        }
      });
    });

    // Open type/dictionary/aggregation panel
    document.querySelectorAll("button[data-type]").forEach(btn => {
      btn.onclick = (e) => {
        const p = e.currentTarget.getAttribute("data-type");
        CURRENT_TYPE_PATH = p || "";
        document.getElementById("tp_path").value = p;
        const curType = types[p] || "";
        document.getElementById("tp_type").value = curType || "";
        const curAgg = aggrules[p] || "";
        document.getElementById("tp_agg").value = curAgg || "";
        PANEL_ROWS = JSON.parse(JSON.stringify((objrules[p] || [])));
        renderDictRows();
        toggleDictBox();
        const panel = document.getElementById("typePanel");
        panel.style.display = "block"; panel.setAttribute("aria-hidden","false");
      };
    });

    // Static edit/clear/hide
    document.querySelectorAll("button[data-edit]").forEach(btn => {
      btn.onclick = (e) => {
        const p = e.currentTarget.getAttribute("data-edit");
        let cur = rules[p] || {mode:"static", static_type:(types[p]||"string"), static_value:""};
        let v = prompt("Static value (string / JSON for object/array):",
                       (cur.static_type==="object"||cur.static_type==="array") ? JSON.stringify(cur.static_value) : String(cur.static_value ?? ""));
        if (v === null) return;
        let t = types[p] || cur.static_type || "string";
        rules[p] = { mode:"static", static_type:t, static_value: tryParseStatic(v, t) };
        render(); push(true);
      };
    });

    document.querySelectorAll("button[data-clear]").forEach(btn => {
      btn.onclick = (e) => {
        const p = e.currentTarget.getAttribute("data-clear");
        delete rules[p];
        render(); push(true);
      };
    });

    document.querySelectorAll("button[data-remove]").forEach(btn => {
      btn.onclick = (e) => {
        const p = e.currentTarget.getAttribute("data-remove");
        hidePath(p);
      };
    });
  }

  // Dictionary builder helpers
  function toggleDictBox(){
    const v = (document.getElementById("tp_type").value || "");
    const box = document.getElementById("dictBox");
    if (v==="dictionary" || v==="object") box.style.display = "block"; else box.style.display = "none";
  }
  document.getElementById("tp_type").onchange = toggleDictBox;

  function renderDictRows(){
    const root = document.getElementById("dictRows"); root.innerHTML = "";
    if (!Array.isArray(PANEL_ROWS)) PANEL_ROWS = [];
    PANEL_ROWS.forEach((row, idx) => {
      const wrap = document.createElement("div");
      wrap.className = "dict-row"; wrap.dataset.idx = String(idx);
      wrap.innerHTML =
        '<input class="k" placeholder="key" value="'+esc(row.key||'')+'"/>' +
        '<select class="mode"><option value="csv">csv</option><option value="static">static</option></select>' +
        '<select class="csv"></select>' +
        '<select class="stype"><option>string</option><option>number</option><option>integer</option><option>boolean</option><option>null</option><option>object</option><option>array</option></select>' +
        '<input class="sval" placeholder="static value" value="'+esc((row.static_value===undefined||row.static_value===null)?'':String(row.static_value))+'"/>' +
        '<select class="astype"><option value="">(inherit)</option><option>string</option><option>number</option><option>integer</option><option>boolean</option><option>null</option><option>object</option><option>array</option></select>' +
        '<button class="delbtn">🗑</button>';
      root.appendChild(wrap);

      // populate selects
      wrap.querySelector(".mode").value = row.mode || "csv";
      const csvSel = wrap.querySelector(".csv");
      csvSel.innerHTML = '<option value=""></option>' + (CSV_COLS||[]).map(c=>'<option value="'+esc(c)+'">'+esc(c)+'</option>').join('');
      csvSel.value = row.csv_col || "";
      wrap.querySelector(".stype").value = row.static_type || "string";
      wrap.querySelector(".astype").value = row.as_type || "";

      // change handlers
      wrap.querySelector(".k").oninput = e => PANEL_ROWS[idx].key = e.target.value;
      wrap.querySelector(".mode").onchange = e => PANEL_ROWS[idx].mode = e.target.value;
      csvSel.onchange = e => PANEL_ROWS[idx].csv_col = e.target.value;
      wrap.querySelector(".stype").onchange = e => PANEL_ROWS[idx].static_type = e.target.value;
      wrap.querySelector(".sval").oninput = e => PANEL_ROWS[idx].static_value = e.target.value;
      wrap.querySelector(".astype").onchange = e => PANEL_ROWS[idx].as_type = e.target.value;
      wrap.querySelector(".delbtn").onclick = () => { PANEL_ROWS.splice(idx,1); renderDictRows(); };
    });
  }

  document.getElementById("addRow").onclick = () => {
    PANEL_ROWS.push({key:"", mode:"csv", csv_col:"", static_type:"string", static_value:"", as_type:""});
    renderDictRows();
  };

  // Toolbar
  document.getElementById("apply").onclick   = () => { push(true); };
  document.getElementById("clearAll").onclick   = () => { rules = {}; render(); push(true); };
  document.getElementById("restoreAll").onclick = () => { hidden = []; renderPathsList(); render(); push(true); };
  document.getElementById("copy").onclick = async () => {
    const text = document.getElementById("stateOut").value;
    try { await navigator.clipboard.writeText(text); alert("State JSON copied!"); }
    catch(e) { alert("Clipboard blocked. Please copy from the box."); }
  };

  // Branch manager
  document.getElementById("filter").oninput = renderPathsList;
  document.getElementById("hide").onclick = () => { const sel = document.getElementById("paths"); const ps = Array.from(sel.selectedOptions).map(o => o.value); ps.forEach(hidePath); };
  document.getElementById("unhide").onclick = () => { const sel = document.getElementById("paths"); const ps = Array.from(sel.selectedOptions).map(o => o.value); unhidePaths(ps); };
  document.getElementById("exclude").onclick = () => { const sel = document.getElementById("paths"); const ps = Array.from(sel.selectedOptions).map(o => o.value); excludePaths(ps); };
  document.getElementById("unexclude").onclick = () => { const sel = document.getElementById("paths"); const ps = Array.from(sel.selectedOptions).map(o => o.value); unexcludePaths(ps); };

  // Type panel buttons
  document.getElementById("tp_close").onclick = () => {
    const panel = document.getElementById("typePanel");
    panel.style.display = "none"; panel.setAttribute("aria-hidden","true");
  };
  document.getElementById("tp_save").onclick = () => {
    let t = (document.getElementById("tp_type").value || "").trim();
    const agg = (document.getElementById("tp_agg").value || "").trim();
    const useDict = (t==="dictionary" || t==="object");
    if (!t) { delete types[CURRENT_TYPE_PATH]; }
    else {
      if (useDict) t = "object";
      if (["string","number","integer","boolean","null","object","array"].includes(t)) types[CURRENT_TYPE_PATH] = t;
      else { alert("Invalid type selected."); return; }
    }
    if (!agg) delete aggrules[CURRENT_TYPE_PATH];
    else if (["unique","list","dictlist"].includes(agg)) aggrules[CURRENT_TYPE_PATH] = agg;
    else { alert("Invalid aggregation."); return; }

    if (useDict){
      const cleaned = (PANEL_ROWS||[]).filter(r => (r.key||"").trim() !== "").map(r => ({
        key: r.key || "",
        mode: (r.mode==="static") ? "static" : "csv",
        csv_col: r.csv_col || "",
        static_type: r.static_type || "string",
        static_value: r.static_value ?? "",
        as_type: r.as_type || ""
      }));
      if (cleaned.length) objrules[CURRENT_TYPE_PATH] = cleaned;
      else delete objrules[CURRENT_TYPE_PATH];
    } else {
      delete objrules[CURRENT_TYPE_PATH];
    }

    const panel = document.getElementById("typePanel");
    panel.style.display = "none"; panel.setAttribute("aria-hidden","true");
    render(); push(true);
  };

  // init
  renderCols();
  renderPathsList();
  render();
  push();
</script>
</body>
</html>
    """

    returned = components.html(
        html.replace("__COLS__",      JS_COLS)
            .replace("__TOKENIZED__", JS_TOKENIZED)
            .replace("__TOKEN_MAP__", JS_TOKEN_MAP)
            .replace("__RULES__",     JS_RULES)
            .replace("__HIDDEN__",    JS_HIDDEN)
            .replace("__EXCLUDED__",  JS_EXCLUDED)
            .replace("__BRANCHES__",  JS_BRANCHES)
            .replace("__TYPES__",     JS_TYPES)
            .replace("__OBJRULES__",  JS_OBJRULES)
            .replace("__AGGRULES__",  JS_AGGRULES),
        height=1000,
        scrolling=True
    )

    if returned:
        try:
            if isinstance(returned, dict):
                return returned
            return json.loads(returned)
        except Exception:
            pass
    return {
        "rules": existing_rules, "hidden": existing_hidden, "excluded": existing_excluded,
        "types": existing_types, "objrules": existing_objrules, "aggrules": existing_aggrules
    }


# =========================== Tab Renderer ===========================
def render_json_mapper_tab(*, prefer_global_csv: bool = True) -> None:
    """
    Render Code 1 as a self-contained tab.
    If prefer_global_csv = True and st.session_state["data_csv"] is a non-empty DataFrame
    (from your Code 2 RLHF viewer), this tab will use it automatically.
    Otherwise it shows its own CSV/XLSX uploader.
    """
    # Style (lightweight)
    st.markdown("""
<style>
main .block-container { max-width: 1200px; }
.small { font-size: 0.9rem; color:#666; }
.codebox { background:#fafafa; border:1px solid #eee; padding:8px; border-radius:6px; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; background:#eef; color:#334; margin-left:6px;}
</style>
""", unsafe_allow_html=True)

    st.title("🧩 JSON Mapper — Grouping, Aggregation & Dictionary Lists")

    # Data sources
    left, right = st.columns(2)
    with left:
        schema_file = st.file_uploader("Upload **Output Schema** (JSON)", type=["json"], key="jm_schema_upl")
    with right:
        df: Optional[pd.DataFrame] = None
        if prefer_global_csv:
            gdf = st.session_state.get("data_csv", None)
            if isinstance(gdf, pd.DataFrame) and not gdf.empty:
                df = gdf
                st.success("Using CSV already loaded in Code 2 (data_csv).")
            else:
                data_file = st.file_uploader("Upload **Data** (CSV/XLSX/XLS)", type=["csv", "xlsx", "xls"], key="jm_data_upl")
                df = jm_read_table(data_file)
        else:
            data_file = st.file_uploader("Upload **Data** (CSV/XLSX/XLS)", type=["csv", "xlsx", "xls"], key="jm_data_upl")
            df = jm_read_table(data_file)

    schema_json = jm_load_json(schema_file)

    if not schema_json:
        st.info("Upload a schema JSON to continue.")
        st.stop()

    # Unwrap schema root
    def unwrap_schema_root(s: dict) -> dict:
        for k in ("outputSchema","schema"):
            if isinstance(s, dict) and isinstance(s.get(k), dict):
                return s[k]
        if isinstance(s, dict) and ("properties" in s or "type" in s or "$schema" in s):
            return s
        node = s
        for k in ("components","schemas"):
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                node = None; break
        if isinstance(node, dict) and node:
            for v in node.values():
                if isinstance(v, dict): return v
        return s

    schema_root = unwrap_schema_root(schema_json)

    # Extract schema paths/types
    leaves_full, typemap_full = jm_gather_schema_paths(schema_root)
    leaves, typemap = jm_prune_array_item_exemplars(leaves_full[:], typemap_full.copy())

    # Build skeleton for visualization
    skeleton = {}
    for p in leaves:
        jm_set_by_path(skeleton, p, jm_type_default(typemap.get(p, "string")))

    # Make EVERY path droppable
    branch_paths = jm_all_branch_paths_from_leaves(leaves)

    # Arrays-of-scalars parents
    arrays_scalar = set()
    for p in leaves_full:
        if p.endswith(".0"):
            parent = p[:-2]
            is_arr_parent = _jm_base_type(typemap_full.get(parent, "")) == "array"
            is_scalar_item = _jm_base_type(typemap_full.get(p, "string")) in ("string","number","integer","boolean")
            if is_arr_parent and is_scalar_item:
                arrays_scalar.add(parent)

    droppable_paths = branch_paths  # all nodes droppable
    csv_cols = sorted(df.columns.tolist()) if (df is not None and not df.empty) else []

    # Session defaults (namespaced)
    for k, v in {
        "jm_rules": {},
        "jm_hidden_paths": [],
        "jm_excluded_paths": [],
        "jm_type_overrides": {},
        "jm_object_rules": {},
        "jm_agg_rules": {},
    }.items():
        if k not in st.session_state: st.session_state[k] = copy.deepcopy(v)

    # Render DnD panel and capture returned state
    state = jm_render_dnd_panel(
        csv_cols=csv_cols,
        skeleton=skeleton,
        drop_paths=droppable_paths,
        typemap=typemap,
        arrays_scalar=arrays_scalar,
        existing_rules=st.session_state["jm_rules"],
        existing_hidden=st.session_state["jm_hidden_paths"],
        existing_excluded=st.session_state["jm_excluded_paths"],
        all_branches=branch_paths,
        existing_types=st.session_state["jm_type_overrides"],
        existing_objrules=st.session_state["jm_object_rules"],
        existing_aggrules=st.session_state["jm_agg_rules"],
    )

    # Persist to session
    st.session_state["jm_rules"] = state.get("rules", {})
    st.session_state["jm_hidden_paths"] = state.get("hidden", [])
    st.session_state["jm_excluded_paths"] = state.get("excluded", [])
    st.session_state["jm_type_overrides"] = state.get("types", {})
    st.session_state["jm_object_rules"] = state.get("objrules", {})
    st.session_state["jm_agg_rules"] = state.get("aggrules", {})

    # -------- Apply Mapping / State JSON (POST-MAPPING) --------
    st.markdown("### Apply Mapping JSON")
    st.caption("Format: `{ \"mapping\": {\"a.b\": \"CSV_Column\"}, \"hidden\": [\"x.y\"], \"excluded\": [\"z\" ] }`")
    mapping_json_text = st.text_area(
        "Paste Mapping JSON",
        height=120,
        key="jm_mapping_area",
        placeholder='{"mapping": {"customer.name": "Name", "customer.age": "Age"}, "hidden": ["meta.debug"], "excluded": ["internal"]}'
    )
    if st.button("✅ Apply Mapping JSON", key="jm_apply_mapping"):
        try:
            parsed = json.loads(mapping_json_text or "{}")
            mapping = parsed.get("mapping", {}) or {}
            hidden  = parsed.get("hidden", []) or []
            excluded = parsed.get("excluded", []) or []
            new_rules = dict(st.session_state["jm_rules"])
            for p, col in mapping.items():
                new_rules[p] = {"mode": "csv", "csv_col": col}
            st.session_state["jm_rules"] = new_rules
            st.session_state["jm_hidden_paths"] = list(hidden)
            st.session_state["jm_excluded_paths"] = list(excluded)
            st.success("Applied Mapping JSON to rules/hidden/excluded. The DnD panel reflects this now.")
        except Exception as e:
            st.error(f"Could not parse Mapping JSON: {e}")

    with st.expander("🧪 (Optional) Apply FULL State JSON (advanced)"):
        st.caption("Format: `{ rules, hidden, excluded, types, objrules, aggrules }` — exactly what the left panel copies.")
        state_json_text = st.text_area(
            "Paste State JSON",
            height=140,
            key="jm_state_area",
            placeholder='{"rules": {...}, "hidden": [...], "excluded": [...], "types": {...}, "objrules": {...}, "aggrules": {...}}'
        )
        if st.button("📥 Apply Full State JSON", key="jm_apply_state"):
            try:
                parsed = json.loads(state_json_text or "{}")
                st.session_state["jm_rules"] = parsed.get("rules", {}) or {}
                st.session_state["jm_hidden_paths"] = parsed.get("hidden", []) or []
                st.session_state["jm_excluded_paths"] = parsed.get("excluded", []) or []
                st.session_state["jm_type_overrides"] = parsed.get("types", {}) or {}
                st.session_state["jm_object_rules"] = parsed.get("objrules", {}) or {}
                st.session_state["jm_agg_rules"] = parsed.get("aggrules", {}) or {}
                st.success("Applied full state. The DnD panel and preview are updated.")
            except Exception as e:
                st.error(f"Could not parse State JSON: {e}")

    # -------- Grouping controls --------
    st.markdown("### Group & Aggregate")
    if df is not None and not df.empty:
        group_cols = st.multiselect("Group by column(s)", options=csv_cols, key="jm_group_by_cols")
    else:
        group_cols = []

    emit_only_mapped = st.checkbox("Emit only mapped fields (recommended)", value=True,
                                   key="jm_emit_only_mapped",
                                   help="If ON, output only fields from your CSV/static/dict rules (plus explicit hidden=null). Excluded branches are removed entirely.")

    # ---------- Live Preview ----------
    rules_live    = st.session_state["jm_rules"]
    hidden_live   = set(st.session_state["jm_hidden_paths"])
    excluded_live = set(st.session_state["jm_excluded_paths"])
    types_live    = st.session_state["jm_type_overrides"]
    objrules_live = st.session_state["jm_object_rules"]
    agg_rules     = st.session_state["jm_agg_rules"]

    def hidden_any(path: str) -> bool:
        return any(path == h or path.startswith(h + ".") for h in hidden_live)
    def excluded_any(path: str) -> bool:
        return any(path == e or path.startswith(e + ".") for e in excluded_live)

    def build_value_for_path(row: Dict[str, Any], p: str) -> Any:
        if p in objrules_live:
            built = {}
            for ent in objrules_live.get(p, []):
                key = (ent.get("key") or "").strip()
                if not key: continue
                mode = ent.get("mode", "csv")
                as_type = (ent.get("as_type") or "").strip()
                if mode == "csv":
                    col = ent.get("csv_col", "")
                    v = row.get(col, "")
                else:
                    stype = ent.get("static_type", "string")
                    sval = ent.get("static_value", "")
                    sval_str = sval if isinstance(sval, str) else json.dumps(sval)
                    v = jm_parse_static(sval_str, stype)
                if as_type:
                    v = jm_coerce(v, as_type)
                built[key] = v
            return built

        rule = rules_live.get(p)
        if not rule:
            return None

        tgt_type = jm_target_type_for(p, typemap, types_live)
        mode = rule.get("mode")

        if mode == "csv":
            col = rule.get("csv_col", "")
            if not col: return None
            val = row.get(col, "")

            if tgt_type == "object":
                try:
                    objv = json.loads(str(val))
                    return objv if isinstance(objv, dict) else {}
                except:
                    return {}
            elif tgt_type == "array":
                try:
                    arrv = json.loads(str(val))
                    return arrv if isinstance(arrv, list) else [arrv]
                except:
                    if isinstance(val, str) and "," in val:
                        return [x.strip() for x in val.split(",") if x.strip()]
                    return [val] if val != "" else []
            elif tgt_type == "null":
                return None
            else:
                return jm_coerce(val, tgt_type)

        elif mode == "static":
            stype = types_live.get(p, rule.get("static_type", "string"))
            sval  = rule.get("static_value", "")
            sval_str = sval if isinstance(sval, str) else json.dumps(sval)
            return jm_parse_static(sval_str, stype)

        return None

    def aggregate_group(rows: pd.DataFrame) -> dict:
        base = {} if emit_only_mapped else copy.deepcopy(skeleton)
        paths = set(rules_live.keys()) | set(objrules_live.keys())

        for p in sorted(paths, key=lambda s: (s.count("."), s)):
            if excluded_any(p):
                continue
            agg_mode = agg_rules.get(p, "unique")

            vals: List[Any] = []
            for _, r in rows.iterrows():
                v = build_value_for_path(r.to_dict(), p)
                vals.append(v)

            if agg_mode == "dictlist":
                out = []
                for v in vals:
                    if isinstance(v, dict):
                        out.append(v)
                    else:
                        if isinstance(v, str):
                            try:
                                dv = json.loads(v)
                                if isinstance(dv, dict):
                                    out.append(dv); continue
                            except:
                                pass
                        out.append({"value": v})
                jm_set_by_path(base, p, out)

            elif agg_mode == "list":
                out = []
                for v in vals:
                    if v is None:
                        continue
                    if isinstance(v, list):
                        out.extend(v)
                    else:
                        out.append(v)
                jm_set_by_path(base, p, out)

            else:  # unique
                chosen = None
                for v in vals:
                    if v is None: 
                        continue
                    if isinstance(v, (list, dict)) and len(v) == 0:
                        continue
                    if isinstance(v, str) and v == "":
                        continue
                    chosen = v
                    break
                jm_set_by_path(base, p, chosen)

        for h in hidden_live:
            try: jm_set_by_path(base, h, None)
            except: pass
        for e in sorted(excluded_live, key=lambda x: x.count("."), reverse=True):
            jm_delete_by_path(base, e)
        return base

    def materialize_row(row: Dict[str, Any]) -> dict:
        base = {} if emit_only_mapped else copy.deepcopy(skeleton)

        for p, entries in (objrules_live or {}).items():
            if excluded_any(p) or hidden_any(p):
                continue
            built = build_value_for_path(row, p)
            jm_set_by_path(base, p, built)

        for p, rule in rules_live.items():
            if excluded_any(p) or hidden_any(p):
                continue
            if p in (objrules_live or {}):
                continue
            val = build_value_for_path(row, p)
            jm_set_by_path(base, p, val)

        for h in hidden_live:
            try: jm_set_by_path(base, h, None)
            except: pass
        for e in sorted(excluded_live, key=lambda x: x.count("."), reverse=True):
            jm_delete_by_path(base, e)
        return base

    # LIVE PREVIEW
    if df is not None and not df.empty and st.session_state.get("jm_group_by_cols"):
        preview_key = None
        preview_rows = None
        for key, gdf in df.groupby(st.session_state["jm_group_by_cols"], dropna=False):
            preview_key = key
            preview_rows = gdf
            break
        if preview_rows is not None:
            preview_obj = aggregate_group(preview_rows)
            st.caption(f"Previewing first group by {st.session_state['jm_group_by_cols']}: {preview_key}")
            st.code(json.dumps(preview_obj, ensure_ascii=False, indent=2), language="json")
        else:
            st.info("No groups were found.")
    else:
        row0 = df.iloc[0].to_dict() if (df is not None and not df.empty) else {}
        preview_obj = materialize_row(row0)
        st.caption("Previewing first row (no grouping)")
        st.code(json.dumps(preview_obj, ensure_ascii=False, indent=2), language="json")

    # ---------- Generate Deliverables ----------
    st.markdown("### Generate Deliverables")
    fname = st.text_input("Output filename", value="deliverables.json", key="jm_output_name")
    if st.button("💾 Build & Download", type="primary", key="jm_build_download"):
        filled: List[dict] = []

        if df is None or df.empty:
            filled.append(preview_obj)
        else:
            if st.session_state.get("jm_group_by_cols"):
                for _, gdf in df.groupby(st.session_state["jm_group_by_cols"], dropna=False):
                    filled.append(aggregate_group(gdf))
            else:
                for _, r in df.iterrows():
                    filled.append(materialize_row(r.to_dict()))

        payload = json.dumps(filled, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ Download JSON", data=payload, file_name=fname or "deliverables.json", mime="application/json", key="jm_dl_btn")
        st.success(f"Prepared {len(filled)} item(s).")
