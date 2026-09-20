# XLA-NPU 本地构建 & 运行指南

## 环境信息

| 项目 | 值 |
|------|-----|
| 操作系统 | openEuler 24.03 LTS（用 `dnf` 安装软件） |
| Python | 3.11.6（`/usr/bin/python3`） |
| CANN | 8.5.0（`/usr/local/Ascend/cann/`） |
| NPU 芯片 | Ascend910B1（通过 `npu-smi info` 查看） |
| 代理工具 | clash-for-linux（`/data/clash-for-linux-master/`） |

## 完整安装步骤

### 步骤 1：安装系统依赖

```bash
dnf install -y xz
```

### 步骤 2：确保网络代理可用

Bazel 编译需要从 GitHub 下载大量依赖，国内环境需要代理。

```bash
# 开启代理
clashon

# 设置终端代理环境变量（Bazel 会读取这些变量）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 验证代理是否生效（期望返回 200/301/302）
curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 https://github.com
```

> **经验**：如果 `curl` 返回 `000`，说明代理端口未监听或 Clash 未正常启动。检查：
> - `ps aux | grep mihomo` 确认进程存在
> - `cat /proc/net/tcp6 | grep 1ED2` 确认 7890 端口监听
> - 尝试 `clashoff && clashon` 重启

### 步骤 3：Source CANN 环境

```bash
source /usr/local/Ascend/cann/set_env.sh
```

> **必须**在运行 `build.sh` 之前执行，否则编译时找不到 CANN 库。

### 步骤 4：运行 build.sh 编译

```bash
cd /data/xla-npu-master/build
bash build.sh install
```

脚本会自动完成以下工作：
1. 下载并安装 JDK 21 → `build/env/java/`
2. 下载并安装 Bazel 7.7.0 → `build/env/bazel/`
3. 克隆 XLA 代码 → `build/code/xla/`
4. 创建 `xla_npu` 软链接到 XLA 仓库内
5. Bazel 编译 PJRT 插件 → `build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so`

脚本支持**断点续传**：如果中途失败，重新执行 `bash build.sh install` 会跳过已完成的步骤。

> **经验**：
> - 首次编译耗时 30 分钟~1 小时+，主要是 Bazel 下载依赖
> - 如果 OOM，减少并行数：`PARALLEL_JOBS=4 bash build.sh install`
> - 如果强制重新安装：`FORCE_REINSTALL=1 bash build.sh install`

### 步骤 5：验证编译产物

```bash
ls -lh /data/xla-npu-master/build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
# 期望输出约 137M 的 .so 文件
```

### 步骤 6：加载运行期环境变量

```bash
source /data/xla-npu-master/build/xla_npu_env
export SOC_VERSION=Ascend910B1
```

`xla_npu_env` 会自动设置：
- `JAVA_HOME` / `PATH`（Java 和 Bazel）
- `XLA_NPU_ROOT_DIR`
- `ASCEND_MLIR_PYTHON_PATH`
- `NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR`

`SOC_VERSION` 需手动设置，根据实际 NPU 芯片型号：
- `Ascend910B1` — Atlas A2 训练系列
- `Ascend910B4` — Atlas A3 训练系列
- 其他型号通过 `npu-smi info` 查看

> **重要经验**：CANN 8.5.0 版本使用 afir 融合后端时，**必须**设置 `USE_OLD_CANN_STYLE=true`，否则编译图时会报 protobuf 解析错误：
> ```bash
> export USE_OLD_CANN_STYLE=true
> ```
> 错误表现为：`Error parsing text-format ascendc_ir.proto.AscGraphDef: Expected integer, got: "1"`

### 步骤 7：安装 JAX

```bash
pip install "jax[cpu]==0.8.1" -i https://mirrors.huaweicloud.com/repository/pypi/simple --trusted-host mirrors.huaweicloud.com
```

> 使用华为云镜像源加速下载。

### 步骤 8：运行 JAX NPU 快速验证

```python
import os
import jax
import jax.numpy as jnp
import numpy as np

# 设置 PJRT 插件路径
PLATFORM_NAME = 'npu'
LIBRARY_PATH = '/data/xla-npu-master/build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so'
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"
os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

# 触发后端发现
import jax.lib.xla_client

# 检查设备
print(f"JAX Default Backend: {jax.default_backend()}")
print(f"Available Devices: {jax.devices()}")

# 运行计算
x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
func = jax.jit(lambda x: jax.nn.gelu(x))
y = func(x)
result = y.block_until_ready()
print(f"Result: {result}")
```

**期望输出**：
```
JAX Default Backend: npu
Available Devices: [NpuDevice(id=0)]
Result: [[0.84... 1.54... 2.99... 4.00...]
 [4.99... 6.00... 6.99... 8.00...]]
```

### 步骤 9（可选）：运行样例

```bash
# 算子测试
cd /data/xla-npu-master/examples/ops
python3 main.py

# MMOE 模型（需先安装 flax）
pip install flax -i https://mirrors.huaweicloud.com/repository/pypi/simple --trusted-host mirrors.huaweicloud.com
cd /data/xla-npu-master/examples/mmoe
python3 main.py --init
```

---

## 一键运行脚本

每次使用 NPU 后端前，需要加载环境变量。建议保存为脚本：

```bash
#!/bin/bash
# 文件: /data/xla-npu-master/setup_npu_env.sh
source /usr/local/Ascend/cann/set_env.sh
source /data/xla-npu-master/build/xla_npu_env
export SOC_VERSION=Ascend910B1
export USE_OLD_CANN_STYLE=true
```

使用方式：
```bash
source /data/xla-npu-master/setup_npu_env.sh
python3 your_script.py
```

---

## 踩坑记录

### 1. Bazel 编译下载依赖超时

**现象**：`Error downloading [...] Connect timed out`

**原因**：GitHub 在国内无法直接访问

**解决**：
- 确保 Clash 代理启动并端口监听
- 设置 `http_proxy` / `https_proxy` 环境变量
- Bazel 会读取这些环境变量走代理下载依赖

### 2. CANN 8.5.0 afir 后端 protobuf 解析错误

**现象**：
```
Error parsing text-format ascendc_ir.proto.AscGraphDef: 6:12: Expected integer, got: "1"
TypeError: HintGraph DeserializeFromReadable fail
```

**原因**：CANN 8.5.0 及之前版本的 afir 接口与新格式不兼容

**解决**：
```bash
export USE_OLD_CANN_STYLE=true
```

> 此环境变量在 README 的"可选配置"中提及，但 CANN 8.5.0 下实际为**必须**设置。

### 3. 缺少 xz 工具

**现象**：`build.sh` 报错 `缺少必要工具: xz`

**解决**：`dnf install -y xz`（openEuler 用 `dnf`，不是 `apt-get`）

### 4. 编译 OOM

**现象**：`java.lang.OutOfMemoryError: Java heap space`

**解决**：减少并行编译任务数
```bash
PARALLEL_JOBS=4 bash build.sh install
```

### 5. ge 后端缺少 Python 依赖

**现象**：`INTERNAL: Failed to initialize GE.`

**解决**：仅在使用 `--npu-auto-fuse-backend=3`（ge 后端）时需要：
```bash
pip install decorator==5.2.1 sympy==1.14.0 attrs==25.4.0
```

---

## 编译选项说明

`build.sh install` 支持通过 `--npu-auto-fuse-backend` 参数选择后端：

| 值 | 后端 | 说明 |
|----|------|------|
| 0 | 无融合 | 仅使用 aclnn 执行 |
| 1（默认） | afir | Auto Fuse IR 后端，需设置 `USE_OLD_CANN_STYLE=true`（CANN ≤ 8.5.0） |
| 3 | ge | Graph Engine 后端，需额外安装 Python 依赖 |
