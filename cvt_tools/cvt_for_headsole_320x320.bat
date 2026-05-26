chcp 65001
set "MODEL_PATH=D:\head_sole\ultralytics\runs\detect\train3\weights\"
set "MODEL_NAME=best"

REM cvt_from_pt_v3.py is configured for 320x320 export.
REM python cvt_from_pt_v3.py "%MODEL_PATH%%MODEL_NAME%.pt"

set "U31_SRC_PATH=..\v3_8bit"
set "NETWORK_NAME=%MODEL_PATH%%MODEL_NAME%.network.csv"
set "WEIGHTS_NAME=%MODEL_PATH%%MODEL_NAME%.weights.txt"
set "MAXMIN_NAME=%MODEL_PATH%%MODEL_NAME%.maxmin.txt"
set "QUANT_NAME=%U31_SRC_PATH%\quant_param_headsole.upinc"
set "U31_MODEL_BIN=headsole_320x320.bin"
set "WFLASH_NAME=%U31_SRC_PATH%\weights_flash_headsole.upinc"
set "INPUT_IMG_PATH=D:\head_sole\data\dataset_2026_05_26\images"

.\cvt_for_u31.exe "%NETWORK_NAME%" "%WEIGHTS_NAME%" "%INPUT_IMG_PATH%" "%MAXMIN_NAME%" "%QUANT_NAME%" "%U31_MODEL_BIN%" "%WFLASH_NAME%"
