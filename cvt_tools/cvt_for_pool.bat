chcp 65001
set "MODEL_PATH_PY=..\\UpixNetwork\\model\\"
set "MODEL_PATH=..\UpixNetwork\model"
set "MODEL_NAME=Pool_320x160_4classes"

python cvt_from_pt.py "%MODEL_PATH_PY%%MODEL_NAME%.pt"

set "U31_SRC_PATH=..\v4_8bit"
set "NETWORK_NAME=%MODEL_PATH%\%MODEL_NAME%.network.csv"
set "WEIGHTS_NAME=%MODEL_PATH%\%MODEL_NAME%.weights.txt"
set "MAXMIN_NAME=%MODEL_PATH%\%MODEL_NAME%.maxmin.txt"
set "QUANT_NAME=%U31_SRC_PATH%\quant_param_pool4class.upinc"
set "U31_MODEL_BIN=%MODEL_NAME%.bin"
set "WFLASH_NAME=%U31_SRC_PATH%\weights_flash_pool4class.upinc"
set "INPUT_IMG_PATH=\\192.168.1.57\u31-ai人脸人行\01-泳池数据\PoolData_V4\images\val"

.\cvt_for_u31.exe "%NETWORK_NAME%" "%WEIGHTS_NAME%" "%INPUT_IMG_PATH%" "%MAXMIN_NAME%" "%QUANT_NAME%" "%U31_MODEL_BIN%" "%WFLASH_NAME%"

REM set "PORJECT=%U31_SRC_PATH%cmos\yolo_cmos_3901_ota.upproj"
REM .\cmd_upasm.exe u31.csv "%U31_SRC_PATH%" "%PORJECT%"
REM 
REM for /f "delims=" %%i in ('python update_version.py') do set NEWVERSION=%%i
REM 
REM .\%U31_SRC_PATH%release.bat "v%NEWVERSION%"
REM 
REM set "DSTDIR=dir_pool\"
REM REM 拷贝文件
REM if not exist %DSTDIR% (
REM     mkdir %DSTDIR%
REM )
REM 
REM xcopy %U31_SRC_PATH%out\v%NEWVERSION%.zip %DSTDIR%
REM xcopy %U31_MODEL_BIN% %DSTDIR%
REM xcopy %NETWORK_NAME% %DSTDIR%
REM xcopy %WEIGHTS_NAME% %DSTDIR%
REM xcopy %MAXMIN_NAME% %DSTDIR%