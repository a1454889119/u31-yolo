@echo off
cd %~dp0

set AppBinPath=".\out\yolo_cmos_3901.bin"
set ModelBinPath="..\cvt_tools\Pool_320x160_4classes.bin"
set WholeBinPath=".\out\app_whole.bin"
set OFFSET_SIZE=0x44000
set WHOLE_SIZE=0x80000
set AppStartAddr=0x4400

@REM typedef struct
@REM {
@REM     uint32_t appStartAddr; //app起始地址 如下：0x4400
@REM     uint32_t appSize;      //app大小
@REM     uint32_t appChecksum;  //app检验和
@REM     uint32_t bOffset;      //参数起始地址
@REM     uint32_t bSize;        //参数大小
@REM     uint32_t bChecksum;    //参数检验和
@REM } FW_INFO;

.\ztools\upix_package.exe %AppBinPath% %ModelBinPath% %WholeBinPath% %OFFSET_SIZE% %WHOLE_SIZE% %AppStartAddr%

@REM @REM 计算MD5值并保存到文件
certutil -hashfile %WholeBinPath% MD5 | findstr /v ":" | findstr /v "CertUtil" > out\filemd5.txt

@REM del out\app.bin 
@REM del out\update.bin
@REM del out\update_00_8K.bin

@REM echo. 输出以一个空行
echo.
REM 检查是否有参数
if "%~1"=="" (
    set "FOLDER_NAME=TMP"
) else (
    set "FOLDER_NAME=%~1"
)

@REM 创建新文件夹
if not exist ".\out\%FOLDER_NAME%" (
    mkdir ".\out\%FOLDER_NAME%"
    echo [INFO] mkdir: ".\out\%FOLDER_NAME%"
) else (
    echo [INFO] dir exist: ".\out\%FOLDER_NAME%"
)

@REM 移动当前目录下的所有 .bin 文件
echo [INFO] move files...
move /Y %WholeBinPath% ".\out\%FOLDER_NAME%" >nul 2>&1
@REM copy /Y %AppBinPath% ".\out\%FOLDER_NAME%" >nul 2>&1
@REM copy /Y %ModelBinPath% ".\out\%FOLDER_NAME%" >nul 2>&1
move /Y ".\out\filemd5.txt" ".\out\%FOLDER_NAME%" >nul 2>&1
copy "README.txt" ".\out\%FOLDER_NAME%" >nul 2>&1

if "%FOLDER_NAME%" NEQ "TMP" (
    echo.
    echo compress to zip...
    powershell -Command "Compress-Archive -Path '.\out\%FOLDER_NAME%' -DestinationPath '.\out\%FOLDER_NAME%.zip' -Force"
    echo done.
)
