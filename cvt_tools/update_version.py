#
# 更新菲亚兰德版本号(版本号中的日期, 最低的版本号+1), 如果不满足要求,请手动修改代码
#
from datetime import datetime
from parse import parse

now = datetime.now()
dateNum = (now.year%100)*10000 + now.month*100 + now.day
filename = "../v4_8bit/memory.upasm"

with open(filename, "r+") as f:
    while True:
        # 1. 記錄當前行起始位置
        start_pos = f.tell()
        
        line = f.readline()
        if not line:
            break
            
        # 2. 判斷邏輯
        if line.startswith('.text "YOLO_FYLD_'):
            #读取版本号
            parseRes = parse('.text "YOLO_FYLD_{:d}_V{:d}.{:d}.{:d}"\n', line)
            # 3. 將指針移回行首
            f.seek(start_pos)
            
            # 4. 覆蓋內容（注意：新內容長度必須與原行一致，否則會破壞後續數據）
            # 如果新內容較短，需用空格補齊；如果較長，會覆蓋到下一行

            new_line = '.text "YOLO_FYLD_' + str(dateNum)  + "_V"
            new_line += str(parseRes.fixed[1]) + "."
            new_line += str(parseRes.fixed[2]) + "."
            new_line += str(parseRes.fixed[3] + 1) + '"\n'
            # 打印输出新的版本号
            print(parseRes.fixed[1], ".", parseRes.fixed[2], ".", parseRes.fixed[3] + 1, sep="")
            f.write(new_line)
            
            # 5. 強制刷新緩衝區（確保寫入生效）
            f.flush()
            break
        