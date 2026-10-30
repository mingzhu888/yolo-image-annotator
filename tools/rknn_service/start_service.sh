#!/bin/bash
# 启动 RKNN 推理服务（后台常驻）
cd /home/zql/rknn_svc || exit 1
pkill -f "[r]knn_svc.py" 2>/dev/null
sleep 1
nohup python3 rknn_svc.py --port 8077 > svc.log 2>&1 < /dev/null &
disown
sleep 3
if pgrep -f "[r]knn_svc.py" > /dev/null; then
    echo "started pid=$(pgrep -f '[r]knn_svc.py' | head -1)"
    tail -2 svc.log
else
    echo "start failed"
    tail -10 svc.log 2>&1
fi
