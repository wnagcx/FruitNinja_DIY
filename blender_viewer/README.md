# Blender Viewer

这个目录存放当前项目的 Blender 可视化脚本。

它的目标是：

- 独立于训练主流程
- 从 `outputs/` 读取 `ply`
- 在 Blender 中显示点云模型
- 支持剖切平面观察
- 支持训练时准实时刷新

## 目录说明

- `launch_viewer.py`
  - 在 WSL 中调用 `/mnt/d/blender/blender.exe`
  - 把监看路径、刷新间隔、颜色模式等参数传给 Blender
- `viewer_runtime.py`
  - 在 Blender 内运行
  - 负责读取 `ply`、构建点云、显示颜色、执行剖切和平面刷新

## 常用命令

### 1. 静态查看 `gs_fill.ply`

```bash
python blender_viewer/launch_viewer.py \
  --watch-path ./outputs/gs_fill.ply \
  --poll-seconds 1.0 \
  --blend-path ./blender_viewer/viewer_static.blend \
  --auto-save-blend
```

### 2. 查看训练实时快照 `viewer_latest.ply`

```bash
python blender_viewer/launch_viewer.py \
  --watch-path ./outputs/viewer_latest.ply \
  --poll-seconds 1.0 \
  --blend-path ./blender_viewer/viewer_train_live.blend \
  --auto-save-blend
```

### 3. 启动训练并输出实时可视化快照（最后参数为几次小循环更新一次）

```bash
python train_orange_demo.py \
  --model_path ./config/orange_demo \
  --physics_config ./config/orange_physics.json \
  --output_path ./outputs \
  --white_bg True \
  --train \
  --gs_path ./outputs/gs_fill.ply \
  --gs_ori_path ./orange_raw.ply \
  --model CED \
  --viewer_live \
  --viewer_save_every_views 10
```

## 颜色显示模式

目前支持三个颜色模式：

- `raw`
  - 直接按 `f_dc_0/1/2` 恢复颜色
- `boosted`
  - 在 `raw` 基础上提高颜色对比，更容易观察
- `grayscale_opacity`
  - 用 opacity 的灰度强弱来显示密度和层次

示例：

```bash
python blender_viewer/launch_viewer.py \
  --watch-path ./outputs/gs_fill.ply \
  --color-mode boosted \
  --blend-path ./blender_viewer/viewer_color.blend \
  --auto-save-blend
```

## Blender 中如何控制

- `fruit_model`
  - 当前显示的点云模型
- `slice_plane`
  - 剖切平面

你可以在 Blender 中：

- 选中 `slice_plane`
- 沿 `X/Y/Z` 方向移动
- 或者旋转它

脚本会自动根据这个平面过滤点云，只显示平面一侧的模型。

## 推荐测试顺序

1. 先查看静态 `gs_fill.ply`
2. 确认模型显示、颜色显示、剖切都正常
3. 再启动训练和 `viewer_latest.ply` 的实时监看
