"""
Streamlit app for browsing inference output JSONL files with images.

Allows interactive exploration of model responses, correctness labels,
and side-by-side image displays for qualitative analysis of model outputs.
"""
import json
import glob
from pathlib import Path
from typing import Any, Optional

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]


def list_output_files(root: Path) -> list[str]:
    """List available output files.

    Includes:
    - spatial/output/*/*.jsonl  (e.g., difference/intersection/union)
    - spatial/output/single/*/*.jsonl  (only files inside subfolders under single)
    """
    base = root / "spatial" / "output_wo_gemini"
    patterns = [
        str(base / "*" / "*.jsonl"),                  # difference/intersection/union and (unfiltered) single/*
        str(base / "single" / "*" / "*.jsonl"),     # single: only consider files in subfolders
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    # De-duplicate and sort
    paths = [Path(f) for f in set(files)]
    # Filter out files that are directly under 'single' (we only want subfolders)
    paths = [p for p in paths if not (p.parent.name == "single" and p.parent.parent == base)]
    files = sorted(str(p) for p in paths)
    return files


def derive_input_path(output_path: Path) -> Path:
    """Derive input JSON path from output JSONL path.

    Example:
      spatial/output/difference/gpt_easy_100.jsonl ->
      spatial/data_path/difference/easy_100.json
    """
    # Default task folder is the immediate parent (e.g., difference)
    task = output_path.parent.name
    # For single outputs, the structure is .../output/single/<model>/<file>.jsonl
    # so task should be 'single' instead of the model name.
    if output_path.parent.parent.name == "single":
        task = "single"
    stem = output_path.stem  # e.g., gpt_easy_100
    # Heuristic: split is everything after the first underscore
    parts = stem.split("_", 1)
    split = parts[1] if len(parts) > 1 else stem
    return ROOT / "spatial" / "data_path" / task / f"{split}.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]
    return items


def load_json(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # Some datasets may wrap entries differently; fall back to list
        return []
    except FileNotFoundError:
        return []


def parse_model_choice(val: Any) -> Optional[int]:
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        try:
            return int(val)
        except Exception:
            return None
    return None


def compute_metrics(
    out_items: list[dict[str, Any]],
    in_items: list[dict[str, Any]]
) -> dict[str, Any]:
    n = 0
    correct_n = 0
    wrong_pos_n = 0
    wrong_subj_n = 0

    for i, rec in enumerate(out_items):
        in_rec = in_items[i] if 0 <= i < len(in_items) else {}

        mc = parse_model_choice(rec.get("model choice"))

        cr = in_rec.get("answer_index")
        wp = in_rec.get("wrong_position_index")
        ws = in_rec.get("wrong_subject_index")

        if isinstance(mc, int):
            n += 1

            if isinstance(cr, int) and mc == cr:
                correct_n += 1
            if isinstance(wp, int) and mc == wp:
                wrong_pos_n += 1
            if isinstance(ws, int) and mc == ws:
                wrong_subj_n += 1

    # print(n, correct_n, wrong_pos_n, wrong_subj_n)

    if n == 0:
        return {"n": 0, "null": 0.0, "acc": 0.0, "wrong_pos_ratio": 0.0, "wrong_subj_ratio": 0.0}
    else:
        return {
            "n": n,
            "null": (len(out_items) - n) / len(out_items),
            "acc": correct_n / n,
            "wrong_pos_ratio": wrong_pos_n / n,
            "wrong_subj_ratio": wrong_subj_n / n,
        }


def get_context_images(record: dict[str, Any], fallback: Optional[dict[str, Any]] = None) -> list[str]:
    # Prefer output keys; fall back to input schema
    keys = ["context images", "context_image", "context_images"]
    for k in keys:
        if k in record and isinstance(record[k], list):
            return record[k]
    if fallback is not None:
        for k in keys:
            if k in fallback and isinstance(fallback[k], list):
                return fallback[k]
    return []


def get_options_images(
    record: dict[str, Any],
    fallback: Optional[dict[str, Any]] = None
) -> list[str]:
    if "options" in record and isinstance(record["options"], list):
        return record["options"]
    if fallback is not None and "options" in fallback and isinstance(fallback["options"], list):
        return fallback["options"]
    return []


def image_path(name: str) -> Path:
    # Images live at spatial/data/controlled_images/{name}.jpeg
    return ROOT / "spatial" / "data" / "controlled_images" / f"{name}.jpeg"


def show_context_layout(names: list[str]) -> None:
    if not names:
        st.info("No context images available.")
        return

    # Head: first 2 or 4 images
    head = names[:-1] if len(names) > 1 else names
    # Tail: last one
    tail = names[-1] if len(names) > 1 else None

    if head:
        st.subheader("Context Images")
        cols = st.columns([1, 1, 0.1, 1, 1])
        labels = ["I1", "I2", None, "I3", "I4"]
        for col, lab in zip(cols, labels):
            with col:
                if lab is None:
                    st.markdown("\u200b")
                    continue
                idx = {"I1": 0, "I2": 1, "I3": 2, "I4": 3}[lab]
                if idx < len(head):
                    name = head[idx]
                    p = image_path(name)
                    if p.exists():
                        st.image(str(p), caption=f"{lab}: {name}", use_container_width=True)
                    else:
                        st.warning(f"Missing image: {name}")

    if tail is not None:
        st.subheader("Question:")
        cols_q = st.columns([1, 1, 0.1, 1, 1])
        with cols_q[0]:
            p = image_path(tail)
            if p.exists():
                st.image(str(p), caption=f"I5: {tail}", use_container_width=True)
            else:
                st.warning(f"Missing image: {tail}")


def show_options(
    names: list[str],
    correct_idx: Optional[int],
    model_idx: Optional[int],
    wrong_pos_idx: Optional[int],
    wrong_subj_idx: Optional[int],
) -> None:
    st.subheader("Options")
    if not names:
        st.info("No options available.")
        return

    cols = st.columns(len(names))
    for i, (c, name) in enumerate(zip(cols, names)):
        with c:
            p = image_path(name)
            is_gt = (correct_idx is not None and i == correct_idx)
            is_model = (model_idx is not None and i == model_idx)
            is_wrong_pos = (wrong_pos_idx is not None and i == wrong_pos_idx)
            is_wrong_subj = (wrong_subj_idx is not None and i == wrong_subj_idx)

            if p.exists():
                st.image(str(p), caption=f"Option {i}: {name}", use_container_width=True)
            else:
                st.warning(f"Missing image: {name}")

            # Colored tags
            tags: list[str] = []
            if is_gt:
                tags.append('<span style="color:#2e7d32; font-weight:600;">Ground Truth</span>')
            if is_wrong_pos:
                tags.append('<span style="color:#ef6c00;">Wrong Position</span>')
            if is_wrong_subj:
                tags.append('<span style="color:#6a1b9a;">Wrong Subject</span>')

            if is_model and is_gt:
                tags.append('<span style="color:#2e7d32; font-weight:600;">(Model Choice)</span>')
            elif is_model and not is_gt:
                tags.append('<span style="color:#c62828; font-weight:600;">(Model Choice)</span>')

            if tags:
                st.markdown(" ".join(tags), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Spatial Visualization", layout="wide")
    st.title("Spatial Analogy Visualizer")

    available = list_output_files(ROOT)
    default_path = str(ROOT / "spatial" / "output" / "difference" / "gpt_easy_100.jsonl")
    default_index = available.index(default_path) if default_path in available else 0

    # Sidebar: file selection (bold title)
    st.sidebar.markdown("**Output JSONL**")
    output_file = st.sidebar.selectbox(
        " ", options=available, index=default_index if available else 0, label_visibility="collapsed"
    )

    if not output_file:
        st.info("No output files found under spatial/output/**.jsonl")
        return

    output_path = Path(output_file)
    input_path = derive_input_path(output_path)

    # Load data
    out_items = load_jsonl(output_path)
    in_items = load_json(input_path)

    if not out_items:
        st.error(f"No items loaded from {output_path}")
        return

    st.sidebar.write(f"**Task:** `{output_path.parent.name}`")
    st.sidebar.write(f"**Input:** `{input_path.relative_to(ROOT) if input_path.exists() else 'Not found'}`")

    # Sidebar: metrics for selected file
    metrics = compute_metrics(out_items, in_items)
    st.sidebar.markdown("**Metrics (selected file)**")

    cols_m_1 = st.sidebar.columns(2)
    cols_m_2 = st.sidebar.columns(2)

    with cols_m_1[0]:
        st.metric("Accuracy", value=f"{metrics['acc']*100:.1f}%")
    with cols_m_1[1]:
        st.metric("Null Rate", value=f"{metrics['null']*100:.1f}%")
    with cols_m_2[0]:
        st.metric("Wrong Pos", value=f"{metrics['wrong_pos_ratio']*100:.1f}%")
    with cols_m_2[1]:
        st.metric("Wrong Subj", value=f"{metrics['wrong_subj_ratio']*100:.1f}%")

    # Optional: metrics for all files
    with st.sidebar.expander("All Files Metrics"):
        # Helpers to detect task and model
        def detect_task(p: Path) -> str:
            if p.parent.parent and p.parent.parent.name == "single":
                return "single"
            return p.parent.name

        def detect_model(p: Path) -> Optional[str]:
            # Infer from filename prefix before first underscore
            stem = p.stem
            return stem.split("_", 1)[0] if "_" in stem else None

        # Build dynamic selector options
        tasks_set, models_set = set(), set()

        for f in available:
            pf = Path(f)
            tasks_set.add(detect_task(pf))
            m = detect_model(pf)
            if m:
                models_set.add(m)

        task_options = ["All"] + sorted(tasks_set)
        model_options = ["All"] + sorted(models_set)
        sel_task = st.selectbox("Task", task_options, index=0)
        sel_model = st.selectbox("Model", model_options, index=0)

        rows = []
        for f in available:
            op = Path(f)
            tname = detect_task(op)
            mname = detect_model(op)
            if sel_task != "All" and tname != sel_task:
                continue
            if sel_model != "All" and mname != sel_model:
                continue

            ip = derive_input_path(op)
            oi = load_jsonl(op)
            ii = load_json(ip)
            mm = compute_metrics(oi, ii)

            display_name = f"{tname}/{op.stem}"
            row = {
                "file": display_name,
                "acc": f"{mm['acc']*100:.1f}%",
                "null": f"{(1 - mm['n']/len(oi))*100:.1f}%",
                "wrong_pos": f"{mm['wrong_pos_ratio']*100:.1f}%",
                "wrong_subj": f"{mm['wrong_subj_ratio']*100:.1f}%",
            }
            rows.append(row)

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # Subset selector (bold title)
    st.sidebar.markdown("**Subset**")
    subset_mode = st.sidebar.selectbox(
        " ",
        options=[
            "All",
            "Correct",
            "Model chose wrong_position_index",
            "Model chose wrong_subject_index",
        ],
        index=0,
        label_visibility="collapsed",
    )

    eligible_indices: list[int] = []
    for i, rec_i in enumerate(out_items):
        in_rec_i = in_items[i] if 0 <= i < len(in_items) else {}

        cr = in_rec_i.get("answer_index")
        wp = in_rec_i.get("wrong_position_index")
        ws = in_rec_i.get("wrong_subject_index")

        mchoice = rec_i.get("model choice")
        if isinstance(mchoice, float):
            try:
                mchoice = int(mchoice)
            except Exception:
                pass
        if not isinstance(mchoice, int):
            mchoice = None

        if subset_mode == "All":
            eligible_indices.append(i)
        elif subset_mode == "Correct":
            if isinstance(cr, int) and mchoice is not None and mchoice == cr:
                eligible_indices.append(i)
        elif subset_mode == "Model chose wrong_position_index":
            if isinstance(wp, int) and mchoice is not None and mchoice == wp:
                eligible_indices.append(i)
        elif subset_mode == "Model chose wrong_subject_index":
            if isinstance(ws, int) and mchoice is not None and mchoice == ws:
                eligible_indices.append(i)

    if not eligible_indices:
        st.info("No items match the current filter.")
        return

    # Number of items
    st.sidebar.write(f"**Items:** {len(out_items)}")

    # Pagination controls
    st.sidebar.markdown("**Items per page**")
    page_size = st.sidebar.slider(
        " ", 1, max(1, min(60, len(eligible_indices))), 20, label_visibility="collapsed"
    )

    total_pages = (len(eligible_indices) + page_size - 1) // page_size
    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0

    cols_nav = st.sidebar.columns(3)
    with cols_nav[0]:
        if st.button("Prev") and st.session_state.page_idx > 0:
            st.session_state.page_idx -= 1
    with cols_nav[1]:
        st.write(f"Page {st.session_state.page_idx + 1}/{total_pages}")
    with cols_nav[2]:
        if st.button("Next") and st.session_state.page_idx < total_pages - 1:
            st.session_state.page_idx += 1

    start = st.session_state.page_idx * page_size
    end = min(start + page_size, len(eligible_indices))
    page_indices = eligible_indices[start:end]

    for idx in page_indices:
        rec = out_items[idx]
        in_rec = in_items[idx] if 0 <= idx < len(in_items) else None

        st.markdown(f"**Item Index:** {idx}")

        # Correct and model choices
        correct_label = rec.get("label") if isinstance(rec.get("label"), int) else None
        model_choice = parse_model_choice(rec.get("model choice"))

        # Context images
        context_names = get_context_images(rec, in_rec)
        show_context_layout(context_names)

        # Options images with markers
        option_names = get_options_images(rec, in_rec)
        wrong_pos_idx = in_rec.get("wrong_position_index") if isinstance(in_rec, dict) else None
        wrong_subj_idx = in_rec.get("wrong_subject_index") if isinstance(in_rec, dict) else None
        show_options(option_names, correct_label, model_choice, wrong_pos_idx, wrong_subj_idx)

        with st.expander("Model Response / Rationale"):
            st.write(rec.get("response", ""))

        st.divider()


if __name__ == "__main__":
    main()
