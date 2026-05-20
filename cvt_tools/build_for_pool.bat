chcp 65001
set "MODEL_PATH_PY=..\\UpixNetwork\\model\\"
set "MODEL_PATH=..\UpixNetwork\model"
set "MODEL_NAME=Pool_320x160_4classes"

REM python cvt_from_pt.py "%MODEL_PATH_PY%\%MODEL_NAME%.pt"

set "U31_SRC_PATH=..\v4_8bit"
set "NETWORK_NAME=%MODEL_PATH%\%MODEL_NAME%.network.csv"
set "WEIGHTS_NAME=%MODEL_PATH%\%MODEL_NAME%.weights.txt"
set "MAXMIN_NAME=%MODEL_PATH%\%MODEL_NAME%.maxmin.txt"
set "QUANT_NAME=%U31_SRC_PATH%\quant_param_pool4class.upinc"
set "U31_MODEL_BIN=%MODEL_NAME%.bin"
set "WFLASH_NAME=%U31_SRC_PATH%\weights_flash_pool4class.upinc"
set "INPUT_IMG_PATH=\\192.168.1.57\u31-ai人脸人行\01-泳池数据\PoolData_V4\images\val"

.\cvt_for_u31.exe "%NETWORK_NAME%" "%WEIGHTS_NAME%" "%INPUT_IMG_PATH%" "%MAXMIN_NAME%" "%QUANT_NAME%" "%U31_MODEL_BIN%" "%WFLASH_NAME%"

set "PORJECT=%U31_SRC_PATH%\cmos\yolo_cmos_3901_ota.upproj"
.\cmd_upasm.exe "u31.csv" "%U31_SRC_PATH%" "%PORJECT%"

set "NEWVERSION=YOLO_FYLD_V1.1.1_20260413"
.\%U31_SRC_PATH%\release.bat "%NEWVERSION%"

set "DSTDIR=dir_pool\"
REM 拷贝文件
if not exist %DSTDIR% (
    mkdir %DSTDIR%
)

xcopy %U31_SRC_PATH%out\%NEWVERSION%.zip %DSTDIR%
xcopy %U31_MODEL_BIN% %DSTDIR%
xcopy %NETWORK_NAME% %DSTDIR%
xcopy %WEIGHTS_NAME% %DSTDIR%
xcopy %MAXMIN_NAME% %DSTDIR%