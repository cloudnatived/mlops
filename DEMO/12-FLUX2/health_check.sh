#!/bin/bash
# health_check.sh

SERVERS=("localhost:8000" "localhost:8003" "localhost:8006")

for server in "${SERVERS[@]}"; do
    echo "检查服务器: $server"
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://$server/v2/health/ready")
    
    if [ "$response" -eq 200 ]; then
        echo "✅ $server 健康状态正常"
        
        # 检查模型状态
        models=$(curl -s "http://$server/v2/models/flux2" | jq -r '.name')
        if [ "$models" == "flux2" ]; then
            echo "✅ FLUX-2模型已加载"
        else
            echo "❌ FLUX-2模型未加载"
        fi
    else
        echo "❌ $server 健康状态异常"
    fi
done
