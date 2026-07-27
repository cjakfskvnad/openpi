# Visuo-Tactile 模型训练与 LIBERO 测试

本文档记录未来视觉触觉 Autoencoder（AE）训练、完整模型训练、LIBERO success rate 测试，以及未来帧预测结果可视化流程。

以下命令默认从仓库根目录执行：

```bash
cd /home/user/kunlun/muze/openpi
```

当前保留两个完整模型配置：

| 配置 | Action/Future attention |
| --- | --- |
| `pi05_expert_visuotactile_spatiotemporal_libero` | Future tactile 可以读取 Action，Action 不能读取 Future tactile |
| `pi05_expert_visuotactile_spatiotemporal_bidirectional_libero` | Action 和 Future tactile 可以双向读取 |

两者都使用：

- 独立 Action Gemma expert 和 Visuo-Tactile Gemma expert
- 独立 tactile SigLIP encoder
- spatiotemporal future latent patch tokens
- `14 × 14` latent grid、patch size `2`，即每帧 `49` 个 future tokens
- action horizon `10`，即总共 `490` 个 future tokens

## 1. 训练 Future Visuo-Tactile Autoencoder

独立 AE 训练不会构造或加载 PI0/PaliGemma，只训练：

```text
FutureVisuoTactileVisionAutoencoder
```

### 多 GPU 训练

下面示例使用 4 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  scripts/train_future_visuotactile_autoencoder.py \
  --config-name pi05_expert_visuotactile_spatiotemporal_libero \
  --exp-name libero_headcam_spatial_ae \
  --batch-size 256 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --preview-interval 1000 \
  --overwrite
```

AE checkpoint 保存到：

```text
checkpoints/future_visuotactile_autoencoder/
└── libero_headcam_spatial_ae/
    ├── 1000/
    │   ├── autoencoder.safetensors
    │   └── training_state.pt
    ├── 2000/
    └── ...
```

训练过程中生成的 reconstruction preview 位于同一实验目录。

### 恢复 AE 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  scripts/train_future_visuotactile_autoencoder.py \
  --config-name pi05_expert_visuotactile_spatiotemporal_libero \
  --exp-name libero_headcam_spatial_ae \
  --batch-size 256 \
  --num-train-steps 30000 \
  --resume
```

不要同时指定 `--resume` 和 `--overwrite`。

## 2. 训练完整模型

完整模型训练前需要：

1. PI0.5 PyTorch 基础权重：`checkpoints/pytorch/pi05_base/model.safetensors`
2. 上一步训练得到的 AE checkpoint
3. LIBERO normalization stats

训练 checkpoint 的目录格式为：

```text
checkpoints/<config_name>/<exp_name>/<step>/
```

### Future attends Action

```bash
CONFIG=pi05_expert_visuotactile_spatiotemporal_libero
EXP=libero_spatiotemporal
AE_DIR=checkpoints/future_visuotactile_autoencoder/libero_headcam_spatial_ae

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  scripts/train_pytorch.py "$CONFIG" \
  --exp-name "$EXP" \
  --pytorch-autoencoder-weight-path "$AE_DIR" \
  --batch-size 8 \
  --num-train-steps 30000 \
  --save-interval 5000 \
  --overwrite
```

`AE_DIR` 可以指向：

- 具体文件：`.../7000/autoencoder.safetensors`
- 具体 step 目录：`.../7000`
- 包含多个数字 step 子目录的实验目录

指向实验目录时，代码会自动选择最新的有效 `autoencoder.safetensors`。

### Bidirectional

```bash
CONFIG=pi05_expert_visuotactile_spatiotemporal_bidirectional_libero
EXP=libero_spatiotemporal_bidirectional
AE_DIR=checkpoints/future_visuotactile_autoencoder/libero_headcam_spatial_ae

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  scripts/train_pytorch.py "$CONFIG" \
  --exp-name "$EXP" \
  --pytorch-autoencoder-weight-path "$AE_DIR" \
  --batch-size 8 \
  --num-train-steps 30000 \
  --save-interval 5000 \
  --overwrite
```

### 恢复完整模型训练

使用和原训练完全相同的 `CONFIG`、`EXP` 和 AE 路径：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  scripts/train_pytorch.py "$CONFIG" \
  --exp-name "$EXP" \
  --pytorch-autoencoder-weight-path "$AE_DIR" \
  --batch-size 8 \
  --num-train-steps 30000 \
  --save-interval 5000 \
  --resume
```

## 3. LIBERO Success Rate 测试

Success rate 测试需要两个终端：

- 终端 1：在主项目环境中启动 PyTorch policy server
- 终端 2：在 LIBERO 环境中运行 simulator client

假设要测试：

```text
config: pi05_expert_visuotactile_spatiotemporal_libero
checkpoint: checkpoints/pi05_expert_visuotactile_spatiotemporal_libero/libero_spatiotemporal/30000
```

### 终端 1：启动 Policy Server

```bash
cd /home/user/kunlun/muze/openpi

CONFIG=pi05_expert_visuotactile_spatiotemporal_libero
CKPT=checkpoints/pi05_expert_visuotactile_spatiotemporal_libero/libero_spatiotemporal/30000

CUDA_VISIBLE_DEVICES=0 \
TORCHDYNAMO_DISABLE=1 \
PYTHONPATH=src \
uv run python -u scripts/serve_policy.py \
  --env LIBERO \
  --port 8011 \
  policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="$CKPT"
```

测试 bidirectional 模型时，替换：

```bash
CONFIG=pi05_expert_visuotactile_spatiotemporal_bidirectional_libero
CKPT=checkpoints/pi05_expert_visuotactile_spatiotemporal_bidirectional_libero/libero_spatiotemporal_bidirectional/30000
```

### 终端 2：运行 LIBERO

第一次使用时，需要按照 [examples/libero/README.md](examples/libero/README.md) 安装 LIBERO 环境。

```bash
cd /home/user/kunlun/muze/openpi
source examples/libero/.venv/bin/activate

export PYTHONPATH="$PWD/third_party/libero:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1

MUJOCO_GL=egl \
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8011 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.num-parallel-envs 10 \
  --args.no-save-video
```

可选 task suite：

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

如果只测试一个 task，添加：

```bash
--args.task-id 0
```

最终日志会输出：

```text
Current task success rate: ...
Current total success rate: ...
Total success rate: ...
```

需要保存 rollout 视频时，删除：

```bash
--args.no-save-video
```

并可指定输出目录：

```bash
--args.video-out-path data/libero/videos/spatiotemporal
```

## 4. 测试未来帧预测

脚本：

```text
scripts/predict_libero_future_visuotactile.py
```

该脚本会：

1. 从一个 LIBERO 初始状态读取当前相机图像和机器人状态
2. 同时预测 action chunk 和未来视觉触觉帧
3. 在 simulator 中执行预测动作
4. 比较模型预测帧与实际 rollout 帧
5. 保存 comparison video、逐帧图片和 metadata

### Future attends Action 模型

```bash
cd /home/user/kunlun/muze/openpi

CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl \
PYTHONPATH=src:third_party/libero \
uv run python scripts/predict_libero_future_visuotactile.py \
  --args.checkpoint-dir checkpoints/pi05_expert_visuotactile_spatiotemporal_libero/libero_spatiotemporal/30000 \
  --args.config-name pi05_expert_visuotactile_spatiotemporal_libero \
  --args.task-suite-name libero_spatial \
  --args.task-id 0 \
  --args.out-dir data/libero/future_visuotactile/spatiotemporal_task0
```

### Bidirectional 模型

```bash
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl \
PYTHONPATH=src:third_party/libero \
uv run python scripts/predict_libero_future_visuotactile.py \
  --args.checkpoint-dir checkpoints/pi05_expert_visuotactile_spatiotemporal_bidirectional_libero/libero_spatiotemporal_bidirectional/30000 \
  --args.config-name pi05_expert_visuotactile_spatiotemporal_bidirectional_libero \
  --args.task-suite-name libero_spatial \
  --args.task-id 0 \
  --args.out-dir data/libero/future_visuotactile/spatiotemporal_bidirectional_task0
```

默认情况下：

- `rollout_steps=None`：执行完整 action/future-frame horizon
- `stop_on_success=True`：任务成功后提前停止当前 episode
- `video_fps=10`
- `resize_size=224`



每个 episode 输出到：

```text
<out_dir>/episode_00/
├── initial.png
├── pred_00.png
├── actual_00.png
├── pred_future.mp4
├── actual_rollout.mp4
├── comparison.mp4
├── contact_sheet.png
├── comparison_contact_sheet.png
├── executed_actions.npy
├── metadata.json
└── ...
```

其中 `comparison.mp4` 用于比较预测未来帧和 simulator 实际帧，`metadata.json` 记录任务、动作执行步数、reward 和 success。

## 常用路径汇总

```text
AE checkpoints:
  checkpoints/future_visuotactile_autoencoder/<exp_name>/<step>/

完整模型 checkpoints:
  checkpoints/<config_name>/<exp_name>/<step>/

LIBERO success rate 视频:
  data/libero/videos/

未来帧预测结果:
  data/libero/future_visuotactile/
```
