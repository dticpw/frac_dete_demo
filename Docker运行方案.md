# Docker 运行方案

本文档记录在当前项目条件下，如何考虑用 Docker 运行 `frac_dete_demo` 以及 TotalSegmentator `appendicular_bones` 方向。

当前背景：

- 本机项目目录：`E:\PG\fracture_detection\frac_dete_demo`
- DICOM 测试数据目录：`E:\PG\fracture_detection\测试`
- 当前本机运行环境：conda `fracmed`
- GPU：RTX 4060 Laptop GPU
- 已申请到 TotalSegmentator `appendicular_bones` license
- 当前 Windows 本机 `fracmed` 模式下，TotalSegmentator 开放 `total` 任务曾在推理阶段出现 native crash

因此，Docker 的优先目标不是替代当前 Gradio demo，而是先用 Linux 容器绕开 Windows 原生依赖问题，验证 `appendicular_bones` 是否能稳定跑通。

## 1. 推荐路线

建议分三步做。

第一步：只验证 Docker + GPU 是否可用。

第二步：在 Docker 中单独运行 TotalSegmentator `appendicular_bones`，先产出骨结构 mask。

第三步：如果 TotalSegmentator 在 Docker 中稳定，再考虑把 Gradio 项目整体容器化，或让 Gradio 继续在 Windows/`fracmed` 中运行、只把 TotalSegmentator 当成外部容器工具调用。

不建议一开始就把整个项目、nnInteractive、TotalSegmentator、Gradio 全部打进一个大镜像。这样排错面太大。

## 2. 本机需要准备什么

### 2.1 Windows / Docker 条件

需要：

- Windows 10/11；
- 已启用 WSL2；
- 已安装 Docker Desktop；
- Docker Desktop 使用 WSL2 backend；
- NVIDIA Windows 驱动支持 WSL2 GPU；
- Docker 容器内能看到 GPU。

先在 PowerShell 中检查：

```powershell
wsl --status
wsl --update
docker version
docker info
```

检查 Docker GPU：

```powershell
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

如果能看到 RTX 4060 Laptop GPU，说明 Docker GPU 链路基本可用。

如果这里失败，不要继续跑 TotalSegmentator。优先检查：

- Docker Desktop 是否启用 WSL2 backend；
- NVIDIA 驱动是否较新；
- WSL2 是否更新；
- Docker Desktop 是否能访问 GPU。

### 2.2 License 保存原则

不要把 TotalSegmentator license 写进 git。

不要写入：

- `Dockerfile`
- `.env`
- markdown 示例里的真实值
- shell 脚本
- commit history

推荐只在本机命令行中输入一次，或放在不提交的本地文件中。

如果需要保存本地 license 文件，建议路径类似：

```text
E:\PG\fracture_detection\private\totalseg_license.txt
```

并确认该目录不在 git 仓库内。

## 3. 方案 A：只用 Docker 跑 TotalSegmentator

这是当前最推荐的第一步。

项目 Gradio 页面仍然用 `fracmed` 运行：

```powershell
D:/python/anaconda/envs/fracmed/python.exe app.py
```

TotalSegmentator 另开 Docker 容器运行，输出 mask 到项目目录或 `outputs` 目录。

### 3.1 准备输入 NIfTI

当前已有探针脚本可以把 DICOM 导出为 NIfTI：

```powershell
D:/python/anaconda/envs/fracmed/python.exe scripts/probe_totalseg.py --case 1 --task total --device cpu --quality fastest --no-inference
```

如果脚本当前没有 `--no-inference` 参数，则可以临时使用已有 probe 逻辑导出 NIfTI，或者后续单独加一个 `scripts/export_case_nifti.py`。

目标是得到类似：

```text
outputs\totalseg_probe\case_1_input.nii.gz
```

如果不想改脚本，也可以让 Docker 容器直接挂载 `E:\PG\fracture_detection\测试` 后在容器内做 DICOM -> NIfTI，但排错会更复杂。

### 3.2 拉取 TotalSegmentator 镜像

官方 Docker 镜像较大，可能超过 20GB。

```powershell
docker pull wasserth/totalsegmentator:latest
```

如果 `latest` 出现版本不可控问题，可以改用固定版本标签。

### 3.3 在容器中配置 license

建议把容器的 `/root` 映射到本地一个不提交的目录，这样 license 和模型缓存可以复用：

```powershell
mkdir E:\PG\fracture_detection\docker_cache\totalseg_home
```

设置 license：

```powershell
docker run --rm -it `
  --gpus all `
  -v "E:\PG\fracture_detection\docker_cache\totalseg_home:/root" `
  wasserth/totalsegmentator:latest `
  totalseg_set_license -l 你的license
```

注意：不要把真实 license 写进文档或提交。

### 3.4 运行 appendicular_bones

假设输入文件在：

```text
E:\PG\fracture_detection\frac_dete_demo\outputs\totalseg_probe\case_1_input.nii.gz
```

输出目录：

```text
E:\PG\fracture_detection\frac_dete_demo\outputs\totalseg_docker\case_1_appendicular_bones
```

运行：

```powershell
mkdir E:\PG\fracture_detection\frac_dete_demo\outputs\totalseg_docker\case_1_appendicular_bones

docker run --rm -it `
  --gpus all `
  -v "E:\PG\fracture_detection\docker_cache\totalseg_home:/root" `
  -v "E:\PG\fracture_detection\frac_dete_demo:/workspace/frac_dete_demo" `
  wasserth/totalsegmentator:latest `
  TotalSegmentator `
    -i /workspace/frac_dete_demo/outputs/totalseg_probe/case_1_input.nii.gz `
    -o /workspace/frac_dete_demo/outputs/totalseg_docker/case_1_appendicular_bones `
    -ta appendicular_bones `
    -d gpu `
    --fastest
```

如果 GPU 报错，可先用 CPU 验证 license 和任务能否启动：

```powershell
docker run --rm -it `
  -v "E:\PG\fracture_detection\docker_cache\totalseg_home:/root" `
  -v "E:\PG\fracture_detection\frac_dete_demo:/workspace/frac_dete_demo" `
  wasserth/totalsegmentator:latest `
  TotalSegmentator `
    -i /workspace/frac_dete_demo/outputs/totalseg_probe/case_1_input.nii.gz `
    -o /workspace/frac_dete_demo/outputs/totalseg_docker/case_1_appendicular_bones_cpu `
    -ta appendicular_bones `
    -d cpu `
    --fastest
```

CPU 会慢很多，但适合先确认 license 和任务参数。

### 3.5 预期输出

`appendicular_bones` 预期会输出多个骨结构 mask，例如：

```text
radius.nii.gz
ulna.nii.gz
carpal.nii.gz
metacarpal.nii.gz
tarsal.nii.gz
metatarsal.nii.gz
...
```

具体输出取决于扫描部位和模型任务定义。

如果是手腕 CT，重点看：

- `radius`
- `ulna`
- `carpal`
- `metacarpal`
- `phalanges_hand`

如果是足部 CT，重点看：

- `tibia`
- `fibula`
- `tarsal`
- `metatarsal`
- `phalanges_feet`

## 4. 方案 B：把 Gradio 项目整体放进 Docker

这个方案可以做，但不是第一优先。

它适合：

- 需要把 demo 发给别人复现；
- 需要固定 Python/package 版本；
- 希望完全隔离 conda 环境；
- 后续要部署到 Linux 服务器。

### 4.1 需要准备 Dockerfile

可以新增类似：

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      gradio \
      pydicom \
      SimpleITK \
      nibabel \
      scipy \
      scikit-image \
      opencv-python-headless \
      pandas \
      matplotlib \
      plotly \
      nnInteractive \
      TotalSegmentator

EXPOSE 7860

CMD ["python", "app.py"]
```

这个 Dockerfile 只是方向草案，不建议直接提交为最终版本。实际要根据 `fracmed` 中的版本锁定依赖，否则后续容易出现版本漂移。

### 4.2 运行项目容器

如果 `app.py` 仍绑定：

```python
server_name="127.0.0.1"
```

容器外无法访问。容器模式下应改成：

```python
server_name="0.0.0.0"
```

或者新增一个单独启动脚本，例如 `app_docker.py`。

运行方式大致为：

```powershell
docker build -t frac-dete-demo:local .

docker run --rm -it `
  --gpus all `
  -p 7860:7860 `
  -v "E:\PG\fracture_detection\测试:/workspace/test_data:ro" `
  -v "E:\PG\fracture_detection\frac_dete_demo\data\models:/app/data/models" `
  -v "E:\PG\fracture_detection\frac_dete_demo\outputs:/app/outputs" `
  frac-dete-demo:local
```

但当前代码中 `config.DATA_DIR = PROJECT_ROOT / "测试"`，在容器内路径会变化。因此整体容器化前，最好先把数据路径改成可由环境变量控制：

```text
FRACTURE_DATA_DIR=/workspace/test_data
```

否则容器内会找不到 `../测试`。

## 5. 当前模式 vs Docker 模式

### 5.1 当前 `fracmed` 本机模式优点

- 已经跑通 Gradio、nnInteractive、启发式候选和 3D 预览；
- 启动简单；
- 调试方便；
- 直接访问 Windows 文件路径；
- 不需要构建大镜像；
- 对日常页面开发最省事。

### 5.2 当前 `fracmed` 本机模式缺点

- TotalSegmentator 在 Windows 推理阶段曾出现 native crash；
- 医学影像大包会污染本机 conda 环境；
- 复现依赖较难；
- 换机器部署时需要重新配置 conda、CUDA、包版本；
- Windows 下部分 Linux 优先的医学工具链更容易出问题。

### 5.3 Docker 模式优点

- Linux 容器环境更贴近 TotalSegmentator / nnU-Net 的主要运行生态；
- 可以绕开 Windows 原生库崩溃问题；
- 依赖隔离更好；
- 复现性更强；
- 后续上服务器更顺；
- 可以单独把 TotalSegmentator 作为外部推理工具，不影响主 Gradio demo。

### 5.4 Docker 模式缺点

- 首次安装 Docker / WSL2 / GPU 支持需要时间；
- 镜像很大，TotalSegmentator 镜像可能超过 20GB；
- Windows 路径挂载和容器路径映射容易出错；
- GPU、CUDA、Docker Desktop、WSL2 之间可能有兼容问题；
- 调试 Gradio 页面不如本机直接运行方便；
- license、模型缓存、输出目录需要单独规划；
- 医院现场演示时，如果 Docker Desktop 或 WSL2 状态异常，恢复成本比本机 conda 更高。

## 6. 建议的实际使用策略

短期建议：

```text
Gradio demo / nnInteractive / 页面开发：继续使用 fracmed
TotalSegmentator appendicular_bones：优先用 Docker 单独验证
```

也就是：

- 主页面仍运行在 Windows + `fracmed`；
- Docker 只负责跑 TotalSegmentator；
- TotalSegmentator 输出的 mask 文件落到 `outputs/totalseg_docker/`；
- 后续由 Python 读取这些 mask 并接入页面展示。

中期建议：

- 如果 Docker 中的 `appendicular_bones` 稳定，新增一个脚本：

```text
scripts/probe_totalseg_docker.ps1
```

用于一键运行 Docker TotalSegmentator。

- 再新增一个 adapter：

```text
AppendicularBonesSegmentationAdapter
```

它只读取 Docker 产出的 mask，不在 Gradio 请求线程里直接启动大型 Docker 推理。

长期建议：

- 如果医院或服务器环境需要统一部署，再考虑整体 Docker 化；
- 整体 Docker 化前，应先做依赖锁定和数据路径环境变量化。

## 7. 推荐下一步

现在你已经拿到 license，建议下一步不是马上改 Gradio，而是先完成最小 Docker 验证：

1. 安装/打开 Docker Desktop；
2. 确认 Docker 使用 WSL2 backend；
3. 运行 `nvidia-smi` 容器测试 GPU；
4. 拉取 `wasserth/totalsegmentator`；
5. 在容器中设置 license；
6. 对 Case 1 的 NIfTI 运行：

```text
appendicular_bones + --fastest + gpu
```

7. 检查是否输出 `radius/ulna/carpal` 或足部相关 mask；
8. 如果成功，再讨论如何把这些 mask 接到页面。

## 8. 参考资料

- Docker Desktop WSL2 GPU 支持说明：https://docs.docker.com/desktop/features/gpu/
- NVIDIA Container Toolkit 安装说明：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NVIDIA CUDA on WSL 用户指南：https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- TotalSegmentator GitHub：https://github.com/wasserth/TotalSegmentator
- TotalSegmentator Dockerfile：https://github.com/wasserth/TotalSegmentator/blob/master/Dockerfile
