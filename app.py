from __future__ import annotations

import gradio as gr

from src import config
from src.annotation_store import save_annotations
from src.candidate_detection import candidate_table, detect_candidates
from src.case_index import case_by_label, case_labels, describe_case, list_cases
from src.dicom_loader import load_case
from src.mesh_preview import build_or_load_mesh
from src.view_rendering import render_views


TABLE_HEADERS = ["candidate_id", "slice_index", "x", "y", "score", "reason", "status"]


def load_selected_case(case_label: str):
    cases = list_cases()
    case = case_by_label(case_label, cases)
    volume_data = load_case(case.case_id, str(case.path))
    candidates = detect_candidates(case.case_id, volume_data.volume_hu)
    table = candidate_table(candidates)
    mesh_path = build_or_load_mesh(case.case_id, volume_data.volume_hu, candidates)
    shape = volume_data.volume_hu.shape

    axial_idx = shape[0] // 2
    coronal_idx = shape[1] // 2
    sagittal_idx = shape[2] // 2
    axial, coronal, sagittal = render_views(
        volume_data.volume_hu,
        axial_idx,
        coronal_idx,
        sagittal_idx,
        candidates,
        config.DEFAULT_WINDOW_CENTER,
        config.DEFAULT_WINDOW_WIDTH,
    )

    info = describe_case(case)
    info += f"\nVolume shape: {shape[0]} x {shape[1]} x {shape[2]}"
    info += "\nCandidate score is heuristic only."

    return (
        {"case_label": case_label, "case_id": case.case_id, "case_path": str(case.path), "candidates": [cand.to_dict() for cand in candidates]},
        info,
        gr.update(maximum=shape[0] - 1, value=axial_idx),
        gr.update(maximum=shape[1] - 1, value=coronal_idx),
        gr.update(maximum=shape[2] - 1, value=sagittal_idx),
        axial,
        coronal,
        sagittal,
        str(mesh_path),
        table,
        "",
        None,
    )


def update_views(state: dict, axial_idx: int, coronal_idx: int, sagittal_idx: int, window_center: int, window_width: int):
    if not state:
        return None, None, None
    volume_data = load_case(state["case_id"], state["case_path"])
    candidates = [_candidate_from_dict(item) for item in state.get("candidates", [])]
    return render_views(volume_data.volume_hu, axial_idx, coronal_idx, sagittal_idx, candidates, window_center, window_width)


def jump_to_candidate(evt: gr.SelectData, state: dict):
    if not state or evt.index is None:
        return gr.update(), gr.update(), gr.update(), "", None
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    candidates = state.get("candidates", [])
    if row_idx is None or row_idx >= len(candidates):
        return gr.update(), gr.update(), gr.update(), "", None
    cand = candidates[row_idx]
    message = f"Selected {cand['candidate_id']} at z={cand['z']}, y={cand['y']}, x={cand['x']}"
    return int(cand["z"]), int(cand["y"]), int(cand["x"]), message, int(row_idx)


def update_candidate_status(state: dict, table_rows, selected_row: int | None, status: str):
    if not state or not table_rows:
        return table_rows, state, "No candidate rows to update."
    rows = _table_to_rows(table_rows)
    if selected_row is None or selected_row >= len(rows):
        return rows, state, "Select a candidate row first."
    rows[selected_row][6] = status
    state["candidates"] = _rows_to_candidate_dicts(state, rows)
    return rows, state, f"Updated {rows[selected_row][0]} to {status}."


def add_manual_candidate(state: dict, table_rows, axial_idx: int, coronal_idx: int, sagittal_idx: int):
    if not state:
        return table_rows, state, "Load a case first."
    rows = _table_to_rows(table_rows)
    case_id = state["case_id"]
    candidate_id = f"{case_id}_manual_{len(rows) + 1:04d}"
    rows.append(
        [
            candidate_id,
            int(axial_idx),
            int(sagittal_idx),
            int(coronal_idx),
            "1.00",
            "manual marker",
            "manual",
        ]
    )
    state["candidates"] = _rows_to_candidate_dicts(state, rows)
    return rows, state, f"Added manual marker {candidate_id}."


def export_annotations(state: dict, table_rows):
    if not state:
        return "Load a case first."
    rows = _table_to_rows(table_rows)
    json_path, csv_path = save_annotations(state["case_id"], rows)
    return f"Saved:\n{json_path}\n{csv_path}"


def _candidate_from_dict(item: dict):
    from src.candidate_detection import Candidate

    return Candidate(
        candidate_id=str(item["candidate_id"]),
        case_id=str(item["case_id"]),
        slice_index=int(item["slice_index"]),
        x=int(item["x"]),
        y=int(item["y"]),
        z=int(item["z"]),
        score=float(item["score"]),
        reason=str(item["reason"]),
        status=str(item.get("status", "unreviewed")),
    )


def _table_to_rows(table_rows) -> list[list]:
    if table_rows is None:
        return []
    if hasattr(table_rows, "values"):
        return table_rows.values.tolist()
    return [list(row) for row in table_rows]


def _rows_to_candidate_dicts(state: dict, rows: list[list]) -> list[dict]:
    records = []
    for row in rows:
        records.append(
            {
                "candidate_id": str(row[0]),
                "case_id": state["case_id"],
                "slice_index": int(row[1]),
                "z": int(row[1]),
                "x": int(row[2]),
                "y": int(row[3]),
                "score": float(row[4]),
                "reason": str(row[5]),
                "status": str(row[6]),
            }
        )
    return records


def build_app() -> gr.Blocks:
    config.ensure_dirs()
    cases = list_cases()
    labels = case_labels(cases)

    with gr.Blocks(title="Fracture Detection Demo") as demo:
        state = gr.State({})
        selected_row = gr.State(None)
        gr.Markdown("# Fracture Detection Demo")
        gr.Markdown("Demo-stage CT DICOM viewer with weak fracture candidate prompts. Candidate scores are heuristic only.")

        with gr.Row():
            case_dropdown = gr.Dropdown(choices=labels, value=labels[0] if labels else None, label="Case", scale=2)
            load_button = gr.Button("Load Case", variant="primary", scale=1)

        case_info = gr.Textbox(label="Case Summary", lines=7)

        with gr.Row():
            axial_slider = gr.Slider(0, 1, value=0, step=1, label="Axial z")
            coronal_slider = gr.Slider(0, 1, value=0, step=1, label="Coronal y")
            sagittal_slider = gr.Slider(0, 1, value=0, step=1, label="Sagittal x")

        with gr.Row():
            window_center = gr.Slider(-500, 1500, value=config.DEFAULT_WINDOW_CENTER, step=10, label="Window Center")
            window_width = gr.Slider(100, 4000, value=config.DEFAULT_WINDOW_WIDTH, step=50, label="Window Width")

        with gr.Row():
            axial_img = gr.Image(label="Axial", type="numpy", height=420)
            coronal_img = gr.Image(label="Coronal", type="numpy", height=420)
            sagittal_img = gr.Image(label="Sagittal", type="numpy", height=420)

        model_3d = gr.Model3D(
            label="3D Bone Preview",
            display_mode="solid",
            clear_color=(1.0, 1.0, 1.0, 1.0),
            height=460,
        )
        gr.Markdown(
            "3D preview is a downsampled bone-surface mesh for spatial orientation only. "
            "Use the three CT views for detail inspection."
        )

        candidates = gr.Dataframe(headers=TABLE_HEADERS, label="Weak Candidates", interactive=True, wrap=True)

        with gr.Row():
            confirm_button = gr.Button("Confirm Selected Candidate")
            reject_button = gr.Button("Reject Selected Candidate")
            manual_button = gr.Button("Add Manual Marker At Current Crosshair")
            export_button = gr.Button("Export Annotations")

        status = gr.Textbox(label="Status", lines=4)

        load_outputs = [
            state,
            case_info,
            axial_slider,
            coronal_slider,
            sagittal_slider,
            axial_img,
            coronal_img,
            sagittal_img,
            model_3d,
            candidates,
            status,
            selected_row,
        ]
        load_button.click(load_selected_case, inputs=[case_dropdown], outputs=load_outputs)
        case_dropdown.change(load_selected_case, inputs=[case_dropdown], outputs=load_outputs)

        view_inputs = [state, axial_slider, coronal_slider, sagittal_slider, window_center, window_width]
        view_outputs = [axial_img, coronal_img, sagittal_img]
        for component in [axial_slider, coronal_slider, sagittal_slider, window_center, window_width]:
            component.change(update_views, inputs=view_inputs, outputs=view_outputs)

        candidates.select(jump_to_candidate, inputs=[state], outputs=[axial_slider, coronal_slider, sagittal_slider, status, selected_row])
        confirm_button.click(lambda s, t, r: update_candidate_status(s, t, r, "confirmed"), inputs=[state, candidates, selected_row], outputs=[candidates, state, status])
        reject_button.click(lambda s, t, r: update_candidate_status(s, t, r, "rejected"), inputs=[state, candidates, selected_row], outputs=[candidates, state, status])
        manual_button.click(
            add_manual_candidate,
            inputs=[state, candidates, axial_slider, coronal_slider, sagittal_slider],
            outputs=[candidates, state, status],
        )
        export_button.click(export_annotations, inputs=[state, candidates], outputs=[status])

        if labels:
            demo.load(load_selected_case, inputs=[case_dropdown], outputs=load_outputs)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
