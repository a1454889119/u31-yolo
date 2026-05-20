chcp 65001
set "MODEL_PATH_PY=..\\UpixNetwork\\model\\"
set "MODEL_PATH=..\UpixNetwork\model\"
set "MODEL_NAME=HeadShoulder_320x160"

REM python cvt_from_pt_v3.py "%MODEL_PATH_PY%%MODEL_NAME%.pt"

set "U31_SRC_PATH=..\v3_8bit"
set "NETWORK_NAME=%MODEL_PATH%%MODEL_NAME%.network.csv"
set "WEIGHTS_NAME=%MODEL_PATH%%MODEL_NAME%.weights.txt"
set "MAXMIN_NAME=%MODEL_PATH%%MODEL_NAME%.maxmin.txt"
set "QUANT_NAME=%U31_SRC_PATH%\quant_param_headshoulder.upinc"
set "U31_MODEL_BIN=%MODEL_NAME%.bin"
set "WFLASH_NAME=.\weights_flash_headshoulder.upinc"
set "INPUT_IMG_PATH=\\192.168.1.57\u31-ai人脸人行\03-开发资料\04-训练验证相关数据\new_datasets\headshoulder_hagrid\images\val"

.\cvt_for_u31.exe "%NETWORK_NAME%" "%WEIGHTS_NAME%" "%INPUT_IMG_PATH%" "%MAXMIN_NAME%" "%QUANT_NAME%" "%U31_MODEL_BIN%" "%WFLASH_NAME%"

REM set "PORJECT=%U31_SRC_PATH%cmos\yolo_cmos_3901.upproj"
REM .\cmd_upasm.exe u31.csv "%U31_SRC_PATH%" "%PORJECT%"

REM set "DSTDIR=dir_facepalm\"
REM 拷贝文件
REM if not exist %DSTDIR% (
REM     mkdir %DSTDIR%
REM )

REM xcopy %U31_SRC_PATH%out\yolo_cmos_3901.bin %DSTDIR%
REM xcopy %U31_MODEL_BIN% %DSTDIR%
REM xcopy %NETWORK_NAME% %DSTDIR%
REM xcopy %WEIGHTS_NAME% %DSTDIR%
REM xcopy %MAXMIN_NAME% %DSTDIR%