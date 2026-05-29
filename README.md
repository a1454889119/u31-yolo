# U31 YOLO 沙包模型转换与测试流程

本文记录 `shabao_192x192.pt` 从 PyTorch 模型转换到 U31 量化文件、再到 C++ 测试工程验证的流程。

## 适用模型约定

| 项目 | 内容 |
|---|---|
| 模型 | `shabao_192x192.pt` |
| 输入尺寸 | `192 x 192` |
| 输入通道 | RGB 三通道 |
| 输入布局 | CHW |
| float 输入范围 | `0~1` |
| int 输入范围 | `0~255` |
| 类别数 | 3 |
| 类别顺序 | `0 red`, `1 blue`, `2 green` |
| 检测输出层 | `43`, `48`, `59`, `64` |

关键约定：

```text
输入：RGB-CHW
输出读取：CWH
float 输入：除以 255
int 输入：不除以 255
bbox：DFL，4 个方向 x 10 bins
```

## 相关文件

| 文件/目录 | 作用 |
|---|---|
| `cvt_tools/cvt_from_pt_v3.py` | `.pt` 导出 ONNX，并生成 `network.csv` / `weights.txt` |
| `cvt_tools/cvt_onnx.py` | ONNX 节点重排和导出入口 |
| `cvt_tools/OnnxModelExport.py` | ONNX 转 UpixNetwork 网络结构和权重 |
| `cvt_tools/v3.order.txt` | ONNX 节点重排顺序 |
| `calib.py` | 从训练集随机复制 1000 张校准图片 |
| `cvt_tools/cvt_for_u31.exe` | 生成 `maxmin.txt`、量化参数、bin、weights upinc |
| `UpixNetwork/model/` | 模型、CSV、权重、maxmin 存放目录 |
| `v3_8bit/quant_param_shabao.upinc` | U31 量化参数 |
| `v3_8bit/weights_flash_shabao.upinc` | U31 权重 flash 地址和长度 |
| `UpixNetwork/test/test_shabao_192x192/` | 沙包模型 C++ 测试工程 |

## 1. `.pt` 转 ONNX / CSV / 权重

进入转换目录：

```bat
cd /d D:\u31-yolo\cvt_tools
```

检查 `cvt_from_pt_v3.py` 里的输入尺寸：

```python
Yolo.export(format='onnx', imgsz=[192, 192])
```

执行转换：

```bat
python cvt_from_pt_v3.py ..\UpixNetwork\model\shabao_192x192.pt
```

成功后应生成：

```text
D:\u31-yolo\UpixNetwork\model\shabao_192x192.onnx
D:\u31-yolo\UpixNetwork\model\shabao_192x192-reorder.onnx
D:\u31-yolo\UpixNetwork\model\shabao_192x192.network.csv
D:\u31-yolo\UpixNetwork\model\shabao_192x192.weights.txt
```

成功日志示例：

```text
ONNX: export success
模型已成功保存到 ..\UpixNetwork\model\shabao_192x192-reorder.onnx，且通过验证
kcount: 30912 bcount: 1238
```

## 2. 准备 maxmin 校准图片

`maxmin.txt` 不能直接由 `.pt` 生成。必须先有：

```text
shabao_192x192.network.csv
shabao_192x192.weights.txt
```

然后使用校准图片跑 float forward，统计每层输出范围。

当前 `calib.py` 里配置的是：

```python
src_dir = Path(r"D:\沙包\datasets7500\images\train")
dst_dir = Path(r"D:\data\shabao_calib")
num_images = 1000
```

如训练集路径不同，先修改 `src_dir`。

从仓库根目录运行：

```bat
cd /d D:\u31-yolo
python .\calib.py
```

生成的校准图片目录：

```text
D:\data\shabao_calib
```

## 3. 生成 maxmin / bin / upinc

进入转换目录：

```bat
cd /d D:\u31-yolo\cvt_tools
```

执行：

```bat
.\cvt_for_u31.exe ^
  "..\UpixNetwork\model\shabao_192x192.network.csv" ^
  "..\UpixNetwork\model\shabao_192x192.weights.txt" ^
  "D:\data\shabao_calib" ^
  "..\UpixNetwork\model\shabao_192x192.maxmin.txt" ^
  "..\v3_8bit\quant_param_shabao.upinc" ^
  "shabao_192x192.bin" ^
  "..\v3_8bit\weights_flash_shabao.upinc"
```

参数含义：

| 参数 | 含义 |
|---|---|
| 1 | 输入 `network.csv` |
| 2 | 输入 `weights.txt` |
| 3 | 校准图片目录 |
| 4 | 输出 `maxmin.txt` |
| 5 | 输出 `quant_param_shabao.upinc` |
| 6 | 输出 `shabao_192x192.bin` |
| 7 | 输出 `weights_flash_shabao.upinc` |

成功后重点检查这些文件：

```text
D:\u31-yolo\UpixNetwork\model\shabao_192x192.maxmin.txt
D:\u31-yolo\v3_8bit\quant_param_shabao.upinc
D:\u31-yolo\cvt_tools\shabao_192x192.bin
D:\u31-yolo\v3_8bit\weights_flash_shabao.upinc
```

## 4. 注意 `cvt_for_u31` 的输入布局

三通道沙包模型必须使用 RGB-CHW 输入。OpenCV 的 `CV_32FC3` 默认是 HWC 交错内存：

```text
RGB RGB RGB ...
```

模型需要的是 CHW 平面内存：

```text
RRRR... GGGG... BBBB...
```

如果 `cvt_for_u31.cpp` 里还存在这种写法，需要特别小心：

```cpp
cv::Mat imgfloat;
img_rgb.convertTo(imgfloat, CV_32FC3);
imgfloat *= 1.0f / 255.0f;
net->forward((float*)imgfloat.data);
```

推荐改成显式 RGB-CHW：

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
            input_chw[0 * 192 * 192 + idx] = row[x][0] / 255.0f;
            input_chw[1 * 192 * 192 + idx] = row[x][1] / 255.0f;
            input_chw[2 * 192 * 192 + idx] = row[x][2] / 255.0f;
        }
    }

    net->forward(input_chw.data());
}
```

如果 maxmin 是用错误输入布局生成的，后面的 int 量化效果会一起偏。

## 5. C++ 测试工程

进入测试工程：

```text
D:\u31-yolo\UpixNetwork\test\test_shabao_192x192
```

打开：

```text
test_shabao_192x192.vcxproj
```

然后在 Visual Studio 中右键生成。

当前测试工程中的核心代码：

```text
UpixNetwork/test/test_shabao_192x192/test_shabao_192x192.cpp
```

它已经包含：

```text
RGB-CHW float 输入
RGB-CHW int 输入
43 / 48 / 59 / 64 层输出读取
DFL decode
NMS
视频检测和区域计数
```

## 6. 测试工程模型路径说明

`test_shabao_192x192.cpp` 中模型路径定义为：

```cpp
#define MODEL_PATH "../../model/"
#define DATA_PATH "../../data/"
```

因此模型文件放哪里，取决于程序运行时的工作目录：

| 运行方式 | `../../model/` 可能指向 |
|---|---|
| Visual Studio 工作目录为项目目录 | `D:\u31-yolo\UpixNetwork\model` |
| 直接从 `x64\Debug` 或 `x64\Release` 运行 exe | `D:\u31-yolo\UpixNetwork\test\test_shabao_192x192\model` |

如果提示加载模型失败，优先检查工作目录和模型文件位置。

测试至少需要这些模型文件：

```text
shabao_192x192.network.csv
shabao_192x192.weights.txt
shabao_192x192.maxmin.txt
```

## 7. 常见问题排查

### `.pt` 导出失败

先确认 `cvt_from_pt_v3.py` 里尺寸是否正确：

```python
imgsz=[192, 192]
```

如果报输入通道错误，优先检查 Ultralytics 导出时 dummy input 是否是 3 通道。

### C++ float 检测框乱飞

优先检查：

```text
1. 输入是否 RGB-CHW
2. 输出是否按 CWH 读取
3. DFL decode 是否和 ONNX 后处理一致
```

### C++ int 效果明显差

优先检查：

```text
1. maxmin.txt 是否用正确 RGB-CHW + /255 输入重新生成
2. 校准图片是否覆盖红/蓝/绿、光照、距离、遮挡
3. int 输入是否 RGB-CHW 且不除 255
4. quant_param_shabao.upinc 和 weights_flash_shabao.upinc 是否是最新生成
```

### 输出读取分不清 WHC / CWH

沙包模型测试代码按 CWH 读取：

```cpp
value = data[ch * w * h + y * w + x];
```

不要写成：

```cpp
value = data[(y * w + x) * c + ch];
```

## 8. 一次完整流程

```text
1. 准备 UpixNetwork/model/shabao_192x192.pt
2. cd /d D:\u31-yolo\cvt_tools
3. python cvt_from_pt_v3.py ..\UpixNetwork\model\shabao_192x192.pt
4. 检查 onnx / reorder.onnx / network.csv / weights.txt 是否生成
5. cd /d D:\u31-yolo
6. 检查 calib.py 的 src_dir / dst_dir
7. python .\calib.py
8. cd /d D:\u31-yolo\cvt_tools
9. 运行 cvt_for_u31.exe 生成 maxmin / bin / upinc
10. 确认 cvt_for_u31 的输入布局是 RGB-CHW
11. 打开 UpixNetwork/test/test_shabao_192x192/test_shabao_192x192.vcxproj
12. 生成并运行测试工程
13. 如果加载模型失败，检查 MODEL_PATH 对应目录
14. 如果检测异常，先对齐 float，再看 int 量化
```
