chcp 65001
set "MODEL_PATH_PY=..\\UpixNetwork\\model\\"
set "MODEL_PATH=..\UpixNetwork\model\"
set "MODEL_NAME=facepalm_320x160_3types"

python cvt_from_pt.py "%MODEL_PATH_PY%%MODEL_NAME%.pt"

set "U31_SRC_PATH=..\v4_8bit\"
set "NETWORK_NAME=%MODEL_PATH%%MODEL_NAME%.network.csv"
set "WEIGHTS_NAME=%MODEL_PATH%%MODEL_NAME%.weights.txt"
set "MAXMIN_NAME=%MODEL_PATH%%MODEL_NAME%.maxmin.txt"
set "QUANT_NAME=%U31_SRC_PATH%quant_param_facepalm3class.upinc"
set "U31_MODEL_BIN=%MODEL_NAME%.bin"
set "WFLASH_NAME=%U31_SRC_PATH%weights_flash_facepalm3class.upinc"
set "INPUT_IMG_PATH=Z:\train_data\facepalm_3types\images\val"

.\cvt_for_u31.exe "%NETWORK_NAME%" "%WEIGHTS_NAME%" "%INPUT_IMG_PATH%" "%MAXMIN_NAME%" "%QUANT_NAME%" "%U31_MODEL_BIN%" "%WFLASH_NAME%"

set "PORJECT=%U31_SRC_PATH%cmos\yolo_cmos_3901.upproj"
.\cmd_upasm.exe u31.csv "%U31_SRC_PATH%" "%PORJECT%"

set "DSTDIR=dir_facepalm\"
REM 拷贝文件
if not exist %DSTDIR% (
    mkdir %DSTDIR%
)

xcopy %U31_SRC_PATH%out\yolo_cmos_3901.bin %DSTDIR%
xcopy %U31_MODEL_BIN% %DSTDIR%
xcopy %NETWORK_NAME% %DSTDIR%
xcopy %WEIGHTS_NAME% %DSTDIR%
xcopy %MAXMIN_NAME% %DSTDIR%