#!/bin/bash
#
# 作者：wangzhen2，2025-07-15，在ubuntu22.04.5上测试通过

# 通过ipmitool或ssh方式收集IPMI信息。使用ipmitool需要安装ipmitool，使用ssh需要安装sshpass
# 安装命令：apt install ipmitool -y 或 apt install sshpass -y

# 确保：本脚本程序、IP列表文件、IPMITOOL.conf、IPMCGET.conf，这4个文件都在当前目录下
# IPMITOOL.conf 每三行记录一个ipmitool参数说明和参数
# IPMCGET.conf 每三行记录一个ipcget参数说明和参数

# 如果每个ip的用户名和密码全部相同：
# 脚本测执行方式为：./IPMI.sh ip地址文件 通用用户名 通用密码，或 bash -x ./IPMI.sh ip地址文件 通用用户名 通用密码（观察执行过程)
# ./IPMI.sh $1 $2 $3 或 bash -x ./IPMI.sh $1 $2 $3

# 如果每个ip的用户名和密码不相同：
# 需要在IP列表文件中每行有3列，第1列为ip地址，第2列为用户名，第3列为密码。脚本测执行方式为：./IPMI.sh ip地址文件 ，或 bash -x ./IPMI.sh ip地址文件（观察执行过程)
# ./IPMI.sh $1 或 bash -x ./IPMI.sh $1

# 重要提示：
# 如果ip地址文件只有ip地址这1列的话，一定要在执行命令时，输入$2(通用用户名)和$3(通用密码)，否则脚本会对每个ip，及每个参数执行20s超时，需要kill -9才能结束脚本
# 如果IP列表文件为excel文件，需要运行先运行：pip3 install csvkit，安装csvkit 处理excel表格。否则会出现提示："未找到in2csv，请使用以下命令安装：pip install csvkit"
# 如果IP列表文件为csv文件，需要运行先运行：apt install dos2unix，安装dos2unix 处理csv表格，否则会出现提示："/usr/bin/ipmitool -I lanplus -U albert -P $'admin\r' -H 10.0.10.236 lan6 print"


# 检查ipmitool是否已安装
if ! command -v ipmitool &> /dev/null; then
    echo "未找到ipmitool，请使用以下命令安装：sudo apt-get install ipmitool"
    exit 1
fi

# 检查sshpass是否已安装
if ! command -v sshpass &> /dev/null; then
    echo "未找到sshpass，请使用以下命令安装：sudo apt-get install sshpass"
    exit 1
fi

# 定义接口类型
I="lanplus"

# 获取脚本参数
IPLIST="$1"
USER="$2"
PASSWORD="$3"
# 定义要执行的命令
IPMITOOL_CMD="timeout 20s /usr/bin/ipmitool -I $I"
IPMCGET_CMD="/usr/bin/sshpass"

# 定义参数说明及参数文件
IPMITOOL_CONF="IPMITOOL.conf"
IPMCGET_CONF="IPMCGET.conf"

# 检查配置文件是否存在
if [ ! -f "$IPMITOOL_CONF" ]; then
    echo "错误：文件 $IPMITOOL_CONF 不存在" >&2
    exit 1
fi

if [ ! -f "$IPMCGET_CONF" ]; then
    echo "错误：文件 $IPMCGET_CONF 不存在" >&2
    exit 1
fi

# 检查IP列表文件是否存在
if [ ! -f "$IPLIST" ]; then
    echo "错误：文件 $IPLIST 不存在" >&2
    exit 1
fi

# 生成唯一的时间戳（精确到小时）
timestamp=$(date +%Y%m%d%H)

# 函数：解析IP地址文件
parse_file() {
    local file="$1"
    local default_user="$2"
    local default_password="$3"
    local ext="${file##*.}"
    
    case "$ext" in
        csv)
	    # 将csv文件从windows格式转换成linux格式
            dos2unix $file
            # 检查列数
            local columns=$(awk -F',' 'NR==1 {print NF}' "$file")
            if [ "$columns" -eq 3 ]; then
                while IFS=',' read -r ip u p _; do
                    if [ -z "$ip" ] || [ -z "$u" ] || [ -z "$p" ]; then
                        echo "警告：跳过空行或不完整的条目: $ip, $u, $p" >&2
                        continue
                    fi
                    if ! [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                        echo "无效的IP地址: $ip" >&2
                        continue
                    fi
                    process_ip "$ip" "$u" "$p"
                done < "$file"
            elif [ "$columns" -eq 1 ]; then
                while IFS=',' read -r ip; do
                    if [ -z "$ip" ]; then
                        echo "警告：跳过空行" >&2
                        continue
                    fi
                    if ! [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                        echo "无效的IP地址: $ip" >&2
                        continue
                    fi
                    process_ip "$ip" "$default_user" "$default_password"
                done < "$file"
            else
                echo "错误：CSV 文件格式不正确，必须有1列或3列" >&2
                exit 1
            fi
            ;;
        xls|xlsx)
            if ! command -v in2csv &> /dev/null; then
                echo "未找到in2csv，请使用以下命令安装：pip install csvkit" >&2
                exit 1
            fi
            local columns=$(in2csv "$file" | awk -F',' 'NR==1 {print NF}' -)
            if [ "$columns" -eq 3 ]; then
                in2csv "$file" | while IFS=',' read -r ip u p _; do
                    if [ -z "$ip" ] || [ -z "$u" ] || [ -z "$p" ]; then
                        echo "警告：跳过空行或不完整的条目: $ip, $u, $p" >&2
                        continue
                    fi
                    if ! [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                        echo "无效的IP地址: $ip" >&2
                        continue
                    fi
                    process_ip "$ip" "$u" "$p"
                done
            elif [ "$columns" -eq 1 ]; then
                in2csv "$file" | while IFS=',' read -r ip; do
                    if [ -z "$ip" ]; then
                        echo "警告：跳过空行" >&2
                        continue
                    fi
                    if ! [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                        echo "无效的IP地址: $ip" >&2
                        continue
                    fi
                    process_ip "$ip" "$default_user" "$default_password"
                done
            else
                echo "错误：XLS/XLSX 文件格式不正确，必须有1列或3列" >&2
                exit 1
            fi
            ;;
        txt|conf)
            local columns=$(awk '{print NF}' "$file" | head -n 1)
            if [ "$columns" -eq 3 ]; then
                while IFS=' ' read -r ip u p; do
                    if [ -z "$ip" ] || [ -z "$u" ] || [ -z "$p" ]; then
                        echo "警告：跳过空行或不完整的条目: $ip, $u, $p" >&2
                        continue
                    fi
                    if ! [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                        echo "无效的IP地址: $ip" >&2
                        continue
                    fi
                    process_ip "$ip" "$u" "$p"
                done < "$file"
            elif [ "$columns" -eq 1 ]; then
                while IFS=' ' read -r ip; do
                    if [ -z "$ip" ]; then
                        echo "警告：跳过空行" >&2
                        continue
                    fi
                    if ! [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                        echo "无效的IP地址: $ip" >&2
                        continue
                    fi
                    process_ip "$ip" "$default_user" "$default_password"
                done < "$file"
            else
                echo "错误：TXT 文件格式不正确，必须有1列或3列" >&2
                exit 1
            fi
            ;;
        *)
            echo "错误：不支持的文件类型: $ext" >&2
            exit 1
            ;;
    esac
}

# 函数：处理单个IP
process_ip() {
    local ip="$1"
    local user="$2"
    local password="$3"
    
    # 更新命令以使用特定的用户名和密码
    local IPMITOOL_LOCAL_CMD="$IPMITOOL_CMD -U $user -P $password -H"
    local IPMCGET_LOCAL_CMD="$IPMCGET_CMD -p $password ssh $user@"
    
    # 判断IP地址的IPMI端口623是否可访问
    timeout 3 bash -c "</dev/tcp/$ip/623" 2>/dev/null
    if [ $? -eq 0 ]; then
        mkdir -p ./IPMILOG/
        result_file="./IPMILOG/${ip}-${timestamp}.log"
        touch "$result_file"
        echo "[+] $ip 的IPMI端口623可连接" >> "$result_file"
        
        echo "开始对 $ip 执行IPMITOOL命令..."
        execute_ipmitool_commands "$ip" "$result_file" "$IPMITOOL_LOCAL_CMD"
        echo "已完成对 $ip 的IPMITOOL命令执行，结果保存在 $result_file"
        
        #echo "开始对 $ip 执行IPMCGET命令..."
        #execute_ipmcget_commands "$ip" "$result_file" "$IPMCGET_LOCAL_CMD"
        #echo "已完成对 $ip 的IPMCGET命令执行，结果保存在 $result_file"
    else
        #echo "[-] $ip 的IPMI端口623无法访问" >&2
        result_file="./IPMILOG/${ip}-${timestamp}.log"
        touch "$result_file"
        echo "[+] $ip 的IPMI端口623无法访问" >> "$result_file"
	
    fi
}

# 函数：执行IPMITOOL命令
execute_ipmitool_commands() {
    local ip="$1"
    local result_file="$2"
    local cmd_prefix="$3"
    
    lines=()
    count=0
    
    while IFS= read -r line; do  
        lines+=("$line")
        ((count++))
        
        if [ $count -eq 3 ]; then
            echo "$ip ${lines[0]}" >> "$result_file"
            echo "$cmd_prefix $ip ${lines[1]}" >> "$result_file"
            echo "##############################" >> "$result_file"
            
            if [[ ${lines[1]} = 'sel list' ]] || [[ ${lines[1]} = 'sel elist' ]]; then
                if ! $cmd_prefix $ip ${lines[1]} | tail -n 20000 >> "$result_file" 2>&1; then
                    echo "错误：执行IPMITOOL命令失败: $ip ${lines[1]}" >> "$result_file"
                fi
            else
                if ! $cmd_prefix $ip ${lines[1]} >> "$result_file" 2>&1; then
                    echo "错误：执行IPMITOOL命令失败: $ip ${lines[1]}" >> "$result_file"
                fi
            fi
            
            echo "##############################" >> "$result_file"            
            lines=()
            count=0
        fi
    done < "$IPMITOOL_CONF"
}

# 函数：执行IPMCGET命令
execute_ipmcget_commands() {
    local ip="$1"
    local result_file="$2"
    local cmd_prefix="$3"
    
    lines=()
    count=0
    
    while IFS= read -r line; do
        lines+=("$line")
        ((count++))
        
        if [ $count -eq 3 ]; then
            echo "$ip ${lines[0]}" >> "$result_file"
            echo "$cmd_prefix$ip ${lines[1]}" >> "$result_file"
            echo "##############################" >> "$result_file"
            
            if ! $cmd_prefix$ip -o StrictHostKeyChecking=no "${lines[1]}" >> "$result_file" 2>&1; then
                echo "错误：执行IPMCGET命令失败: $ip ${lines[1]}" >> "$result_file"
            fi
            
            echo "##############################" >> "$result_file"            
            lines=()
            count=0
        fi
    done < "$IPMCGET_CONF"
}

# 主程序：解析IP列表文件并处理
parse_file "$IPLIST" "$USER" "$PASSWORD"

