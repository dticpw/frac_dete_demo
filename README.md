# Fracture Detection Demo

Minimal local demo for browsing ultra-high-resolution CT DICOM cases and showing weak fracture candidates.

## Run

```powershell
D:/python/anaconda/envs/th123/python.exe app.py
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

- `交互页面说明.md`: explains the page layout, three CT views, candidate IDs such as `1_edge_0492`, score meanings, and annotation workflow.
- `代码文件规划.md`: records the planned code structure and staged implementation route.
