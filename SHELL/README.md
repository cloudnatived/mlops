


```
find ./ -type f |xargs -i sed -i s#tensorflow:1.0.1#tensorflow:1.2.1#g {}



#########################################################################################
#!/bin/sh 
ADDR="http://www.w3school.com.cn"
if [ $1 ]
then
    ADDR=$1
fi
SERVER=${ADDR#http://}
SERVER=${SERVER%%/*}
wget -bc \
    --html-extension \
    --restrict-file-names=windows \
    --convert-links \
    --page-requisites \
    --execute robots=off \
    --mirror \
    --exclude-directories /try \
    --user-agent="Chrome/10.0.648.204" \
    --no-check-certificate \
    --reject "aggregator*" \
    -o $SERVER.log \
    $ADDR 2>&1 &
wait
find $SERVER -type f -name "*.css" -exec cat {} \; |
grep -o 'url(/[^)]*)' |
sort |
uniq |
sed 's/^url(/(.*/))$/http:////'$SERVER'/1/' |
wget --mirror --page-requisites -i -

for i in `find $SERVER -type f -name "*.css"`; do
    PREFIX="$(echo $i | sed 's/[^//]*//g; s///$//; s////../////g')"
    sed -i 's/url(///url('$PREFIX'/g' $i
done


#########################################################################################
#!/bin/bash
url_list=(
https://www.cnblogs.com/paul8339/p/11328459.html
https://blog.51cto.com/u_12790562/3991953
http://192.168.11.128/inde.html
)

function check_url(){
 for url in ${url_list[*]}
   do
      wget -q $url -O /dev/null
      if [ $? -eq 0 ];then
         echo "$url is working"
      else
       echo "$url is not working"
     fi
   done
}


function main(){
  while true
    do
      check_url
      sleep 10
   done
}



#########################################################################################
#!/bin/bash
MATRIX="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LENGTH="9"
while [ "${n:=1}" -le "$LENGTH" ]
do
        PASS="$PASS${MATRIX:$(($RANDOM%${#MATRIX})):1}"
        let n+=1
done
        echo "$PASS"
exit 0


Ip_check.sh
#########################################################################################
#!/bin/bash

# mail -s "Zhen AW Wang" bjwzhen@cn.ibm.com

DATE=`date +\%Y\%m\%d`;
mkdir "$DATE";
CC=`cat IP.conf|grep -v ^#|grep -v ^$`;


CHECK_ON(){
#for C in `cat IP.conf|grep -v ^#|grep -v ^$`
for C in $CC
do
/usr/bin/nmap -sP "$C".0/24 |grep "$C" |awk '{ print $NF }' > "$DATE"/"$C".0.on
#/usr/bin/nmap -sP "$C".0/24 |grep "$C" |awk '{ print $NF }'|xargs -i sed -i s#$NF# ON#g {} > "$DATE"/"$C".0.on
sed -i 's#$# ON#g' "$DATE"/"$C".0.on;
done
}

CHECK_OFF(){
for C in $CC
do
for IP in `seq 1 254`
do
        ST="$C"."$IP";
        STT=`grep "$ST" "$DATE"/"$C"'.0.on'`;

       if [ -z "$STT" ]
        then
                echo "$ST" OFF >> "$DATE"/"$C".0.off;
        fi
done


done
}

CHECK_ON;
CHECK_OFF;


#########################################################################################








```


# AWK

```
参考资料：
http://bbs.chinaunix.net/thread-833305-1-1.html

#心形图形实例
awk 'BEGIN{while(u<20){print sin(u)*sin(v),cos(u)*sin(v+u);v=v+0.01;u=u+0.01}}' |awk -f ColorPlot.awk
#爱心型
awk 'BEGIN{while(u<20){print sin(u)*sin(v),rand()*cos(u)*sin(v+u);v=v+0.01;u=u+0.01}}'|awk -f ColorPlot.awk
#绳结
awk 'BEGIN{while(u<10){print sin(u+v)*sin(v),cos(u+v)*sin(v);v=v+0.01;u=u+0.01}}'|awk -f ColorPlot.awk
#蝴蝶
awk 'BEGIN{while(u<10){print sin(u+v)*sin(v),cos(u)*sin(v);v=v+0.01;u=u+0.01}}'|awk -f ColorPlot.awk
#花瓣
awk 'BEGIN{while(u<10){print sin(u+w)*cos(v)*sin(w+u),sin(u)*sin(v+u)*sin(w);v=v+0.01;u=u+0.01;w=w+0.01}}'|awk -f ColorPlot.awk
#蝙蝠
awk 'BEGIN{while(u<10){print sin(u+w)*cos(v+w)*sin(w+u),sin(u)*sin(v+u)*sin(w);v=v+0.01;u=u+0.01;w=w+0.01}}'|awk -f ColorPlot.awk
#螺旋
awk 'BEGIN{while(w<20){print sin(w)*w,cos(w)*w;w=w+0.01}}'|awk -f ColorPlot.awk
#花蕊
awk 'BEGIN{while(u<10){print sin(u)*cos(v+u)*sin(u+v),cos(v+u)*cos(u)*sin(v+u);v=v+0.01;u=u+0.01}}' | awk -f ColorPlot.awk
#剪刀
awk 'BEGIN{while(u<10){print sin(u+v)*cos(v+u)*sin(u+v),cos(v+u)*cos(u)*sin(v+u);v=v+0.01;u=u+0.01}}'| awk -f ColorPlot.awk
#圆：      
awk 'BEGIN{while(k<10){print sin(k),cos(k);k=k+0.01}}'   |  awk -f ColorPlot.awk
#正弦线:   
awk 'BEGIN{while(k<10){print sin(k),k;k=k+0.01}}'        |  awk -f ColorPlot.awk
#抛物线：  
awk 'BEGIN{k=-10;while(k<10){print k^2,k;k=k+0.01}}'     |  awk -f ColorPlot.awk
#直线：    
paste <(seq 1 0.01 10)  <(seq 1 0.01 10)                 |  awk -f ColorPlot.awk
#圆盘:    
awk 'BEGIN{while(k<100){print sin(k),rand()*cos(k);k=k+0.01}}' |awk -f ColorPlot.awk

花圈:    awk 'BEGIN{srand()
                        while(k++<20000){
                           x=2-3*rand()
                           y=2-4*rand()
                           if(x^2+y^2>0.6&&x^2+y^2<1||x^2+y^2<0.3&&x^2+y^2>0.1)
                              print x,y
                           }
                       }'       | awk -f ColorPlot.awk

awk 'BEGIN{srand() while(k++<20000){x=2-3*rand() y=2-4*rand() if(x^2+y^2>0.6&&x^2+y^2<1||x^2+y^2<0.3&&x^2+y^2>0.1) print x,y }}‘ | awk -f ColorPlot.awk

菱圈:    awk 'BEGIN{srand()
                        while(k++<20000){
                           x=1-2*rand()
                           y=1-2*rand()
                           if(x+y<=1&&x-y<=1&&-x+y<=1&&-x-y<=1&&x^2+y^2>=1/2)
                              print x,y
                           }
                       }'       | awk -f ColorPlot.awk

```




