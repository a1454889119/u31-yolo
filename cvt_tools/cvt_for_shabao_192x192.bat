chcp 65001

set "MODEL_PATH=..\UpixNetwork\model"
set "MODEL_NAME=shabao_192x192"

set "NETWORK_NAME=%MODEL_PATH%\%MODEL_NAME%.network.csv"
set "WEIGHTS_NAME=%MODEL_PATH%\%MODEL_NAME%.weights.txt"
set "MAXMIN_NAME=%MODEL_PATH%\%MODEL_NAME%.maxmin.txt"

set "U31_SRC_PATH=..\v3_8bit"
set "QUANT_NAME=%U31_SRC_PATH%\quant_param_shabao.upinc"
set "U31_MODEL_BIN=%MODEL_NAME%.bin"
set "WFLASH_NAME=%U31_SRC_PATH%\weights_flash_shabao.upinc"

set "INPUT_IMG_PATH=..\UpixNetwork\data\shabao_calib"

.\cvt_for_u31.exe "%NETWORK_NAME%" "%WEIGHTS_NAME%" "%INPUT_IMG_PATH%" "%MAXMIN_NAME%" "%QUANT_NAME%" "%U31_MODEL_BIN%" "%WFLASH_NAME%"

pause