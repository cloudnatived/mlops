#!/bin/bash
# ./check_port.sh 192.168.1.1 80 5
# ip=`cat iplist`; for i in $ip ; do sh 1.sh $i 2198 3 ; done;



# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # 无颜色

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo -e "${RED}错误：请提供IP地址和端口号${NC}"
    echo "用法: $0 IP_ADDRESS PORT [TIMEOUT]"
    exit 1
fi

IP=$1
PORT=$2
TIMEOUT=${3:-2}  # 默认超时时间为2秒

# 检查telnet命令是否存在
if ! command -v telnet &> /dev/null; then
    echo -e "${RED}错误：未找到telnet命令。请先安装telnet。${NC}"
    exit 1
fi

# 测试端口连通性
echo -e "${YELLOW}正在测试 ${IP}:${PORT} 的连通性，超时时间 ${TIMEOUT} 秒...${NC}"

# 使用超时命令限制telnet执行时间
( echo -e "\n" | telnet $IP $PORT 2>&1 ) | timeout $TIMEOUT cat | grep -q "Connected to"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ${IP}:${PORT} 端口可访问${NC}"
    exit 0
else
    echo -e "${RED}✗ ${IP}:${PORT} 端口不可访问或超时${NC}"
    exit 1
fi    
