# LabelPro

基于 [LabelMe](https://github.com/wkentaro/labelme) 开发的图像标注工具，增加了批量关键点管理等实用功能。

## 安装

**要求：Python 3.12 或更高版本。**

### 方式一：使用 uv（推荐，自动管理 Python 版本）

```bash
# 安装 uv（如果还没有）
# Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/ZiChen-Whisper/labelpro.git
cd labelpro

# 安装依赖并运行
uv sync
uv run labelpro
```

### 方式二：使用 pip

```bash
git clone https://github.com/ZiChen-Whisper/labelpro.git
cd labelpro

# 确保 Python >= 3.12
python --version

# 安装
pip install -e .

# 运行
labelpro
```

> 如果 `pip install -e .` 报错，检查 Python 版本是否 >= 3.12。
> 可以用 `conda create -n labelpro python=3.12 -y && conda activate labelpro` 创建新环境。

### 命令行参数

```bash
labelpro                        # 打开 GUI
labelpro image.jpg              # 标注单张图片
labelpro /path/to/images/       # 标注整个文件夹
labelpro image.jpg --output annotations/  # 指定标注输出目录
labelpro image.jpg --labels labels.txt   # 使用预定义标签列表
```

## 新增功能

### 批量删除关键点

菜单：**编辑 → 批量删除关键点…**（快捷键 `Ctrl+Shift+D`）

支持对多个标注文件批量删除指定的关键点（point 类型标注），按 `group_id` 分组进行选择。

**使用方法：**

1. 打开一个文件夹（该文件夹中应包含图片和对应的 JSON 标注文件）
2. 按 `Ctrl+Shift+D` 打开批量删除对话框
3. 左侧文件列表中勾选要处理的文件（默认全选）
4. 选择目标组：按水平位置（从左到右排序），或选择「所有组」
5. 勾选要删除的关键点标签
6. 点击「执行批量删除」确认

**预览导航：**
- `D` — 下一帧
- `A` — 上一帧
- `Ctrl+F` — 适配窗口
- 鼠标滚轮 — 缩放

> 注意：批量删除会直接修改原始 JSON 文件，请提前备份。

## 标注格式

标注保存为 JSON 文件，格式与 LabelMe 兼容：

```json
{
  "version": "5.0.0",
  "flags": {},
  "shapes": [
    {
      "label": "鼻子",
      "points": [[320, 240]],
      "group_id": 1,
      "shape_type": "point",
      "flags": {}
    }
  ],
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

## 许可

[GPL-3.0](LICENSE)

基于 [LabelMe](https://github.com/wkentaro/labelme) 二次开发。
