# U31 YOLO 沙包检测模型转换与 C++ 推理流程

本文档记录从 `shabao_192x192.pt` 到 UpixNetwork / U31 C++ 检测运行的完整流程。

当前仓库相关文件：

```text
cvt_tools/cvt_from_pt_v3.py
UpixNetwork/cvt_for_u31/cvt_for_u31.cpp
UpixNetwork/model/
UpixNetwork/data/
```

---

## 1. 模型约定

| 项目 | 说明 |
|---|---|
| 模型文件 | `shabao_192x192.pt` |
| 输入尺寸 | `192 x 192` |
| 输入通道 | RGB 三通道 |
| 类别数 | 3 |
| 类别顺序 | `0 red`, `1 blue`, `2 green` |
| float 输入 | RGB-CHW，数值范围 `0~1` |
| int 输入 | RGB-CHW，数值范围 `0~255` |
| UpixNetwork 输出读取 | CWH |
| 检测输出层 | `43`, `48`, `59`, `64` |

PyTorch / ONNX 最终输出是：

```text
output0: 1 x 7 x 2880
```

其中：

```text
0: cx
1: cy
2: w
3: h
4: red_score
5: blue_score
6: green_score
```

但转换成 UpixNetwork 后，不是直接输出 `1x7x2880`，而是输出 YOLO head 中间结果：

```text
layer 43: 24 x 24 x 3
layer 48: 24 x 24 x 40
layer 59: 48 x 48 x 3
layer 64: 48 x 48 x 40
```

含义：

```text
3 通道  = class logits
40 通道 = bbox DFL 输出，4 个方向 x 10 bins
```

因此 C++ 端需要自己做：

```text
sigmoid
DFL softmax + expectation
grid + stride decode
NMS
draw boxes
```

---

## 2. 推荐目录结构

```text
C:\u31-yolo\
├── cvt_tools\
│   ├── cvt_from_pt_v3.py
│   ├── cvt_for_u31.exe
│   └── v3.order.txt
│
└── UpixNetwork\
    ├── model\
    │   ├── shabao_192x192.pt
    │   ├── shabao_192x192.onnx
    │   ├── shabao_192x192-reorder.onnx
    │   ├── shabao_192x192.network.csv
    │   ├── shabao_192x192.weights.txt
    │   └── shabao_192x192.maxmin.txt
    │
    └── data\
        ├── shabao_test.jpg
        └── shabao_calib\
```

---

## 3. `.pt` 转 ONNX / network.csv / weights.txt

进入转换目录：

```bat
cd /d C:\u31-yolo\cvt_tools
```

执行：

```bat
python cvt_from_pt_v3.py ..\UpixNetwork\model\shabao_192x192.pt
```

成功后应生成：

```text
C:\u31-yolo\UpixNetwork\model\shabao_192x192.onnx
C:\u31-yolo\UpixNetwork\model\shabao_192x192-reorder.onnx
C:\u31-yolo\UpixNetwork\model\shabao_192x192.network.csv
C:\u31-yolo\UpixNetwork\model\shabao_192x192.weights.txt
```

成功日志示例：

```text
ONNX: export success
模型已成功保存到 ..\UpixNetwork\model\shabao_192x192-reorder.onnx，且通过验证
kcount: 30912 bcount: 1238
```

当前 `cvt_from_pt_v3.py` 的主要流程：

```python
import sys
from ultralytics import YOLO
from cvt_onnx import reorder_convert

ptname = sys.argv[1]

Yolo = YOLO(model=ptname).to('cpu')
Yolo.export(format='onnx', imgsz=[192, 192])

onnxName = ptname.replace('.pt', '.onnx')
reordered_onnxName = onnxName.replace('.onnx', '-reorder.onnx')
csv_name = onnxName.replace('.onnx', '.network.csv')
txt_name = onnxName.replace('.onnx', '.weights.txt')

reorder_convert(
    onnxName,
    "v3.order.txt",
    reordered_onnxName,
    [],
    csv_name,
    txt_name
)
```

如果导出时报错：

```text
expected input to have 3 channels, but got 1 channels instead
```

说明导出时 dummy input 被生成成了单通道。可以尝试把导出改成：

```python
Yolo.export(format='onnx', imgsz=[192, 192], ch=3)
```

如果当前 Ultralytics 不支持 `ch=3` 参数，就需要修改本地 `ultralytics/engine/exporter.py` 里生成 dummy input 的地方，把输入通道从 `1` 改为 `3`。

---

## 4. 验证 `.pt` / `.onnx`

在进入 UpixNetwork C++ 调试之前，建议先验证：

```text
shabao_192x192.pt
shabao_192x192.onnx
shabao_192x192-reorder.onnx
```

如果 `.pt / ONNX / reorder.onnx` 检测正常，而 C++ 检测异常，优先检查：

```text
1. 输入是不是 RGB-CHW
2. 输出是不是按 CWH 读取
3. DFL decode 是否和 ONNX 后处理一致
4. maxmin.txt 是否用同样的 RGB-CHW float 输入生成
```

---

## 5. 准备 maxmin 校准图片

`maxmin.txt` 不能直接由 `.pt` 生成。必须先有：

```text
shabao_192x192.network.csv
shabao_192x192.weights.txt
```

然后使用校准图片跑 float forward，统计每层输出范围。

推荐目录：

```text
C:\u31-yolo\UpixNetwork\data\shabao_calib
```

从训练集随机抽 1000 张图片：

```python
import random
import shutil
from pathlib import Path

src_dir = Path(r"C:\沙包\datasets7500\images\train")
dst_dir = Path(r"C:\u31-yolo\UpixNetwork\data\shabao_calib")

num_images = 1000
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

dst_dir.mkdir(parents=True, exist_ok=True)

images = [
    p for p in src_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_exts
]

selected = random.sample(images, min(num_images, len(images)))

for i, src_path in enumerate(selected, 1):
    dst_path = dst_dir / src_path.name

    if dst_path.exists():
        dst_path = dst_dir / f"calib_{i:04d}_{src_path.name}"

    shutil.copy2(src_path, dst_path)

print(f"copied {len(selected)} images to {dst_dir}")
```

---

## 6. 修改 `cvt_for_u31.cpp` 的校准输入

`maxmin.txt` 的生成必须和 C++ float 推理使用同一种预处理：

```text
BGR image
resize 192x192
BGR -> RGB
RGB -> CHW
float /255
net.forward(float*)
```

三通道模型不能直接把 OpenCV 的 `CV_32FC3` 数据传给 `net->forward()`，因为 OpenCV 的 `CV_32FC3` 是 HWC 交错格式：

```text
RGB RGB RGB ...
```

模型需要 CHW 平面格式：

```text
RRRR... GGGG... BBBB...
```

### 错误写法

```cpp
cv::Mat imgfloat;
img_rgb.convertTo(imgfloat, CV_32FC3);
imgfloat *= 1.0f / 255.0f;

net->forward((float*)imgfloat.data);
```

### 正确写法

把 `UpixNetwork/cvt_for_u31/cvt_for_u31.cpp` 里的 `run_image()` 改成：

```cpp
static void run_image(UpixNetwork* net, cv::Mat& img)
{
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(192, 192));

    cv::Mat img_rgb;

    if (img_resized.channels() == 3) {
        cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);
    }
    else if (img_resized.channels() == 1) {
        cv::cvtColor(img_resized, img_rgb, cv::COLOR_GRAY2RGB);
    }
    else {
        printf("unsupported image channels: %d\n", img_resized.channels());
        return;
    }

    static std::vector<float> input_chw;
    input_chw.resize(192 * 192 * 3);

    for (int y = 0; y < 192; y++) {
        const cv::Vec3b* row = img_rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < 192; x++) {
            int idx = y * 192 + x;

            input_chw[0 * 192 * 192 + idx] = row[x][0] / 255.0f; // R
            input_chw[1 * 192 * 192 + idx] = row[x][1] / 255.0f; // G
            input_chw[2 * 192 * 192 + idx] = row[x][2] / 255.0f; // B
        }
    }

    net->forward(input_chw.data());
}
```

修改后重新编译：

```text
UpixNetwork/cvt_for_u31/cvt_for_u31.exe
```

---

## 7. 生成 maxmin.txt、bin 和 upinc

进入：

```bat
cd /d C:\u31-yolo\cvt_tools
```

执行：

```bat
cvt_for_u31.exe ^
  "..\UpixNetwork\model\shabao_192x192.network.csv" ^
  "..\UpixNetwork\model\shabao_192x192.weights.txt" ^
  "..\UpixNetwork\data\shabao_calib" ^
  "..\UpixNetwork\model\shabao_192x192.maxmin.txt" ^
  "..\v3_8bit\quant_param_shabao.upinc" ^
  "shabao_192x192.bin" ^
  "..\v3_8bit\weights_flash_shabao.upinc"
```

参数含义：

```text
参数1: network.csv
参数2: weights.txt
参数3: 校准图片目录
参数4: 输出 maxmin.txt
参数5: 输出 quant_param_shabao.upinc
参数6: 输出 shabao_192x192.bin
参数7: 输出 weights_flash_shabao.upinc
```

`cvt_for_u31.cpp` 的 `main()` 是 7 参数流程：

```text
networkfilename
weightsfilename
imagepath
maxminfilename
quantfilename
modelbinfilename
weightsupinc
```

它会先执行：

```cpp
dump_maxmin(networkfilename, weightsfilename, imagepath, maxminfilename)
```

然后执行：

```cpp
net.loadQuant(maxminfilename)
dump_quant_params(...)
dump_bin(...)
dump_weights_upinc(...)
```

输出文件：

```text
C:\u31-yolo\UpixNetwork\model\shabao_192x192.maxmin.txt
C:\u31-yolo\v3_8bit\quant_param_shabao.upinc
C:\u31-yolo\cvt_tools\shabao_192x192.bin
C:\u31-yolo\v3_8bit\weights_flash_shabao.upinc
```

运行时会看到类似：

```text
..\UpixNetwork\data\shabao_calib目录下有1000个图片
```

控制台进度使用 `\r` 刷新同一行，不一定每张图都换行显示。

---

## 8. C++ float 推理输入

float 推理输入必须是：

```text
RGB
CHW
0~1
```

示例：

```cpp
static void run_image_float_rgb(UpixNetwork* net, cv::Mat& img_rgb)
{
    static std::vector<float> input_chw;
    input_chw.resize(192 * 192 * 3);

    for (int y = 0; y < 192; y++) {
        const cv::Vec3b* row = img_rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < 192; x++) {
            int idx = y * 192 + x;

            input_chw[0 * 192 * 192 + idx] = row[x][0] / 255.0f; // R
            input_chw[1 * 192 * 192 + idx] = row[x][1] / 255.0f; // G
            input_chw[2 * 192 * 192 + idx] = row[x][2] / 255.0f; // B
        }
    }

    net->forward(input_chw.data());
}
```

---

## 9. C++ int 推理输入

int 推理输入必须是：

```text
RGB
CHW
0~255
```

示例：

```cpp
static void run_image_int_rgb(UpixNetwork* net, cv::Mat& img_rgb)
{
    static std::vector<int> input_chw;
    input_chw.resize(192 * 192 * 3);

    for (int y = 0; y < 192; y++) {
        const cv::Vec3b* row = img_rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < 192; x++) {
            int idx = y * 192 + x;

            input_chw[0 * 192 * 192 + idx] = row[x][0]; // R
            input_chw[1 * 192 * 192 + idx] = row[x][1]; // G
            input_chw[2 * 192 * 192 + idx] = row[x][2]; // B
        }
    }

    net->forward(input_chw.data());
}
```

---

## 10. 输出层与 CWH 读取

沙包模型的 UpixNetwork 输出层：

```text
layer 43: 24 x 24 x 3
layer 48: 24 x 24 x 40
layer 59: 48 x 48 x 3
layer 64: 48 x 48 x 40
```

必须按 CWH 读取：

```cpp
value = data[ch * w * h + y * w + x];
```

不要按 WHC 读取：

```cpp
value = data[(y * w + x) * c + ch];
```

示例：

```cpp
static float get_cwh_f(const float* data, int w, int h, int c, int x, int y, int ch)
{
    return data[ch * w * h + y * w + x];
}

static int get_cwh_i(const int* data, int w, int h, int c, int x, int y, int ch)
{
    return data[ch * w * h + y * w + x];
}
```

---

## 11. DFL Decode

两个尺度：

```text
24 x 24, stride = 8
48 x 48, stride = 4
```

每个尺度：

```text
score: 3 通道，做 sigmoid
bbox : 40 通道，4 个方向，每方向 10 bins
```

bbox 通道含义：

```text
0..9    left
10..19  top
20..29  right
30..39  bottom
```

DFL expectation：

```text
softmax(10 bins)
sum(prob[i] * i)
```

坐标计算：

```text
anchor_x = x + 0.5
anchor_y = y + 0.5

x1 = (anchor_x - left)   * stride
y1 = (anchor_y - top)    * stride
x2 = (anchor_x + right)  * stride
y2 = (anchor_y + bottom) * stride
```

最终做 NMS 并画框。

---

## 12. C++ 加载模型

float：

```cpp
UpixNetwork net_float;

if (!net_float.loadNetwork(MODEL_PATH "shabao_192x192.network.csv")) {
    printf("load float network failed\n");
    return;
}

if (!net_float.loadWeight(MODEL_PATH "shabao_192x192.weights.txt")) {
    printf("load float weight failed\n");
    return;
}
```

int：

```cpp
UpixNetwork net_int;

if (!net_int.loadNetwork(MODEL_PATH "shabao_192x192.network.csv")) {
    printf("load int network failed\n");
    return;
}

if (!net_int.loadWeight(MODEL_PATH "shabao_192x192.weights.txt")) {
    printf("load int weight failed\n");
    return;
}

if (!net_int.loadQuant(MODEL_PATH "shabao_192x192.maxmin.txt")) {
    printf("load quant failed\n");
    return;
}
```

---

## 13. 推荐调试顺序

### 13.1 验证 `.pt`

先确认 `.pt` 在 Python 中检测正常。

### 13.2 验证 ONNX

确认：

```text
shabao_192x192.onnx
shabao_192x192-reorder.onnx
```

都能检测正常。

### 13.3 验证 C++ float

float 必须先对齐 ONNX。  
如果 float 不对，先不要调 int。

重点检查：

```text
1. 输入是否 RGB-CHW
2. 输出是否按 CWH 读取
3. DFL decode 是否正确
```

### 13.4 验证 C++ int

如果 float 正常，int 不正常，重点检查：

```text
1. maxmin.txt 是否用 RGB-CHW + /255 重新生成
2. 校准集是否覆盖红/蓝/绿、光照、距离、遮挡
3. int 输入是否 RGB-CHW 且不除 255
4. int 输出缩放是否和 UpixNetwork 量化一致
```

---

## 14. 常见问题

### 14.1 `.pt` 正常，C++ 框乱飞

通常是输入布局错。

错误：

```cpp
cv::Mat imgfloat;
img_rgb.convertTo(imgfloat, CV_32FC3);
net->forward((float*)imgfloat.data);
```

正确：

```text
RGB -> CHW -> /255 -> forward
```

---

### 14.2 C++ score 很低

通常是输出读取错。

正确：

```cpp
data[ch * w * h + y * w + x]
```

错误：

```cpp
data[(y * w + x) * c + ch]
```

---

### 14.3 旧模型能跑，沙包模型不行

旧模型多半是灰度单通道，HWC / CHW 不容易暴露问题。

沙包模型是 RGB 三通道，必须严格使用：

```text
RGB-CHW 输入
CWH 输出读取
DFL decode
```

旧 `TargetDecoder` 不直接适用于当前沙包模型。

---

## 15. 完整流程总结

```text
1. 准备 shabao_192x192.pt
2. 运行 cvt_from_pt_v3.py
3. 生成 onnx / reorder.onnx / network.csv / weights.txt
4. 准备 shabao_calib 校准图片
5. 修改 cvt_for_u31.cpp，确保校准输入是 RGB-CHW
6. 编译 cvt_for_u31.exe
7. 运行 cvt_for_u31.exe
8. 生成 maxmin.txt / quant_param_shabao.upinc / bin / weights_flash_shabao.upinc
9. C++ 加载 network.csv / weights.txt / maxmin.txt
10. 输入图片转 RGB-CHW
11. 读取 43 / 48 / 59 / 64 输出
12. 按 CWH 读取
13. DFL decode
14. NMS
15. 画框
```

核心原则：

```text
输入：RGB-CHW
输出：CWH
float 输入：0~1
int 输入：0~255
bbox：DFL 4 x 10 bins
旧 TargetDecoder 不直接适用于当前沙包模型
```
