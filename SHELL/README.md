# SHELL

#### shell 十三問

shell 十三問
参考文档：
http://bbs.chinaunix.net/thread-218853-1-1.html

1) 為何叫做 shell ？
http://bbs.chinaunix.net/viewthr ... p;page=2#pid1454336
2) shell prompt(PS1) 與 Carriage Return(CR) 的關係？ (2008-10-30 02:05 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=2#pid1467910
3) 別人 echo、你也 echo ，是問 echo 知多少？( 2008-10-30 02:08 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=3#pid1482452
4) " "(雙引號) 與 ' '(單引號)差在哪？  (2008-10-30 02:07 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=4#pid1511745
5) var=value？export 前後差在哪？ (2008-10-30 02:12 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=5#pid1544391
6) exec 跟 source 差在哪？ (2008-10-30 02:17 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=6#pid1583329
7) ( ) 與 { } 差在哪？
http://bbs.chinaunix.net/viewthr ... p;page=6#pid1595135
8) $(( )) 與 $( ) 還有${ } 差在哪？ (2008-10-30 02:20 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=7#pid1617953
9) $@ 與 $* 差在哪？
http://bbs.chinaunix.net/viewthr ... p;page=7#pid1628522
10) && 與 || 差在哪？ (2008-10-30 02:21 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=7#pid1634118
11) > 與 < 差在哪？ (2008-10-30 02:24 最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=7#pid1636825
12) 你要 if 還是 case 呢？ (2008-10-30 02:25最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=8#pid1679488
13) for what? while 與 until 差在哪？ (2008-10-30 02:26最後更新)
http://bbs.chinaunix.net/viewthr ... p;page=8#pid1692457





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

awk 'BEGIN{srand();while(k++<20000){x=2-3*rand();y=2-4*rand();if(x^2+y^2>0.6&&x^2+y^2<1||x^2+y^2<0.3&&x^2+y^2>0.1) print x,y }}'| awk -f ColorPlot.awk

菱圈:    awk 'BEGIN{srand()
                        while(k++<20000){
                           x=1-2*rand()
                           y=1-2*rand()
                           if(x+y<=1&&x-y<=1&&-x+y<=1&&-x-y<=1&&x^2+y^2>=1/2)
                              print x,y
                           }
                       }'       | awk -f ColorPlot.awk

awk 'BEGIN{srand();while(k++<20000){x=1-2*rand();y=1-2*rand();if(x+y<=1&&x-y<=1&&-x+y<=1&&-x-y<=1&&x^2+y^2>=1/2) print x,y}}'| awk -f ColorPlot.awk
```

#### ColorCat.awk
```
ColorCat.awk
##############
#!/usr/bin/awk
#  Write by dbcat
#  EMail:deeperbluecat@Gmail.com
#  run : awk -f ColorCat.awk YourFile

BEGIN{
        srand()
        }

{
        split($0,Myth,"")
        ColorPrint(Myth,length($0))
}

function ColorPrint(Myth,xlen)
{
   for(i=1;i<=xlen;i++)
    {
       Color="\033[1;"int(31+7*rand())
       printf "%s;3m%s\033[0m",Color,Myth[i]
    }
    printf "\n"
}
##############
```

#### ColorPlot.awk
```
##############
#! /usr/bin/awk
# GAWK彩色作图程序
# 作者: dbcat
# Email: deeperbluecat@Gmail.Com
# 日期: 2006-9-25
# 测试环境: Gawk 3.1.4, bash 3.00.16(1), SUSE 9.3
# 运行方法: awk 'BEGIN{while(k<10){print sin(k),cos(k);k=k+0.01}}' >datafile
#           awk -f ColorPlot.awk datafile

BEGIN{
          srand()
          xlen=35
          ylen=35
          InitGraph(Myth,xlen,ylen)
  }
  
  {
          X_Max=X_Max>$1?X_Max:$1
          X_Min=X_Min<$1?X_Min:$1
          Y_Max=Y_Max>$2?Y_Max:$2
          Y_Min=Y_Min<$2?Y_Min:$2
          X_Label[NR]=$1
          Y_Label[NR]=$2
  }
  
  
END{
          CreateGraph(Myth,NR)
          PrintGraph(Myth)
  }
  
  function InitGraph(Myth,xlen,ylen,i,j)
   {
     for(i=1;i<=xlen;i++)
       for(j=1;j<=ylen;j++)
          Myth[i,j]=" "
   }
  
  function CreateGraph(Myth,Len,i)
   {
  
         for(i=1;i<=Len;i++)
            {
             X_Label[i]=int((X_Label[i]-X_Min)/(X_Max-X_Min)*(xlen-1) + 1)
             Y_Label[i]=int((Y_Label[i]-Y_Min)/(Y_Max-Y_Min)*(ylen-1) + 1)
             Myth[X_Label[i],Y_Label[i]]=int(40+60*rand())
            }
  
   }
  
  function PrintGraph(Myth,i,j)
   {
     for(i=1;i<=xlen;i++)
      {
        for(j=1;j<=ylen;j++)
           {
            color="\033[1;"int(31+7*rand())
            printf " %s;1m%c\033[0m",color,Myth[i,j]
           }
        printf "\n"
      }
   }

##############
```

