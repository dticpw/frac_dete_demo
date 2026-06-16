# Fracture Detection Demo

Minimal local demo for browsing ultra-high-resolution CT DICOM cases and showing weak fracture candidates.

## Run

```powershell
./run_demo_fracmed.ps1
```

Or directly:

```powershell
D:/python/anaconda/envs/fracmed/python.exe app.py
```

The first load of each case reads and stacks the DICOM slices, so it can take a few seconds.

## Scope

This is a demo-stage workflow:

- DICOM case indexing
- CT volume loading
- Axial/coronal/sagittal views
- Downsampled 3D bone-surface preview
- Bone-window rendering
- Lightweight heuristic candidate prompts
- Annotation export

The candidate score is only a heuristic sorting value, not a clinical confidence score.

## Docs

- `医学专用环境说明.md`: records the dedicated `fracmed` conda environment and installed medical-imaging packages.
- `交互页面说明.md`: explains the page layout, three CT views, candidate IDs such as `1_edge_0492`, score meanings, and annotation workflow.
- `启发式算法说明.md`: explains the current rule-based weak candidate algorithm and its limitations.
- `外部模型接口设计.md`: records the reserved adapter interface for future external/reference models.
- `代码文件规划.md`: records the planned code structure and staged implementation route.
