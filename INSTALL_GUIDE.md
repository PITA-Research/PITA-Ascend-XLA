# XLA-NPU 安装指南

## 概述

本文档介绍如何使用 `build.sh` 脚本安装 XLA-NPU。

## 系统要求

### 必须满足
- **Python = 3.11**：必须预先安装（系统自带、apt 安装或 conda 环境）
- **基本工具**：`wget`， `curl`， `git`， `tar`
- **磁盘空间**：至少 10GB 可用空间

### 推荐配置
- 内存：至少 8GB
- CPU：多核处理器（脚本会自动使用所有可用核心）

## 安装前准备

### 1. 安装 Python

#### 方法一：使用 conda（推荐）
```bash
# 创建新环境
conda create -n xla-npu python=3.11
conda activate xla-npu

# 验证安装
python --version
```
#### 方法二：使用系统包管理器
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11 python3-pip

# 验证安装
python3.11 --version
```

### 2. 安装基本工具
```bash
sudo apt-get update
sudo apt-get install wget curl git tar
```

### 3. 克隆 xla-npu 仓库
```bash
git clone https://gitcode.com/cann/xla-npu.git /path/to/xla-npu
```

## 安装步骤

### 1. 进入安装脚本目录
```bash
cd /path/to/xla-npu/build
```

### 2. 给脚本添加执行权限
```bash
chmod +x build.sh
```

### 3. 运行安装脚本
```bash
./build.sh ${action} --npu-auto-fuse-backend=${backend}
```

或者，如果你使用 conda 环境：
```bash
conda activate ${xla-npu}
./build.sh ${action} --npu-auto-fuse-backend=${backend}
```
- ${action}：表示脚本命令，取值如下
  - install：安装，默认不支持重复安装，如果需要重复安装，需要设置环境变量FORCE_REINSTALL为1，如：export FORCE_REINSTALL=1。
  - uninstall：卸载，默认卸载需要二次确认，如果不希望二次确认，需要设置环境变量FORCE_UNINSTALL为1，如：export FORCE_UNINSTALL=1。
  - help：帮助，查看build.sh脚本支持命令的帮助信息。

- --npu-auto-fuse-backend=`${backend}`：表示编译选项，仅${action}为install时支持，可选，`${backend}`取值如下：
  - 0：不支持融合, 仅使用aclnn执行
  - 1（默认值）：使用afir后端
  - 3：使用ge后端

- ${xla-npu}：conda环境名称
- 其它安装环境变量说明：
  - PARALLEL_JOBS：并行编译任务数，默认为cpu核数。可选，如：export PARALLEL_JOBS=4。

### 4. 验证 XLA-NPU 编译产物
```bash
cd /path/to/xla-npu
ls -lh build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
```

### 5. 设置环境变量
```bash
source xla_npu_env
```

## 常见问题

### 1. Python 版本不满足要求

**错误信息**：
```
Python 版本不满足要求 (需要 = 3.11， 当前 3.9.x)
```

**解决方案**：
```bash
# 使用 conda 安装
conda install python=3.11

# 或使用 apt 安装 (Ubuntu 22.04+)
sudo apt-get install python3.11
```

### 2. 编译时内存不足

**错误信息**：
```
java.lang.OutOfMemoryError: Java heap space
```

**解决方案**：

减少并行编译任务数：
```bash
PARALLEL_JOBS=4 ./build.sh install
```

### 3. 权限问题

**错误信息**：
```
无法创建目录: Permission denied
```

**解决方案**：
```bash
# 确保对工作目录有写权限
sudo chown -R $USER:$USER /path/to/xla-npu

# 或使用 sudo 运行脚本（不推荐）
sudo ./build.sh install
```

### 4. ge后端场景运行时缺少依赖

**错误信息**：
```
INTRENAL: Failed to initialize GE.
```

**解决方案**：

安装依赖包：
```bash
pip install decorator==5.2.1
pip install sympy==1.14.0
pip install attrs==25.4.0
```

## 卸载

```bash
cd /path/to/xla-npu/build
./build.sh uninstall
```

---

**注意**: 本安装脚本已针对国内网络环境优化，使用华为镜像源加速下载。如需使用其他镜像源，可修改脚本中的镜像配置。
