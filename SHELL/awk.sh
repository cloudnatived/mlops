


awk -F':' '{if($4>20) {print $NF}}' passwd #命令行


ll *20100423* |awk '{(total+=$5)}; END{print total}'  #求和。


awk -F':' '$4>100' passwd  #第4列大于100的。
awk -F':' '/101/ {print $3 + 10}' passwd #匹配的加上10。输出。
awk -F '[:|]'    '{print $1}' passwd |head #厉害。
awk 'BEGIN { OFS="%"} {print $1,$2}' passwd |head
awk 'BEGIN { FS="[: \t|]" } {print $1,$2,$3}' passwd|head #替换分隔符。
awk -F':' 'BEGIN { OFS="%"} {print $1,$2}' passwd |head #替换分隔符。
awk -F':' '$3 * $4 >100 {print $3,$4}' passwd |head  #乘机大于100的就输出。


# 格3000行插入一行
awk '1; NR % 3000==0 {print "commit;"}' < DG.sql > GG.sql
awk '1; NR % 3000==0 {print "commit;"}' < DG.sql > GG.sql


awk '1; NR % 3000==0 {print "commit;"}' < DG.sql > GG.sql
awk '1; NR % 3000=00 {print "commit;"}' < DG.sql > GG.sql

* pci5             U0.1-P2        PCI Bus
+ scsi0            U0.1-P2/Z1     Wide/Ultra-3 SCSI I/O Controller
+ hdisk0           U0.1-P2/Z1-A8  16 Bit LVD SCSI Disk Drive (36400 MB)
+ hdisk1           U0.1-P2/Z1-A9  16 Bit LVD SCSI Disk Drive (36400 MB)
+ hdisk2           U0.1-P2/Z1-Aa  16 Bit LVD SCSI Disk Drive (36400 MB)
+ hdisk3           U0.1-P2/Z1-Ab  16 Bit LVD SCSI Disk Drive (36400 MB)
* ses0             U0.1-P2/Z1-Af  SCSI Enclosure Services Device
+ scsi1            U0.1-P2/Z2     Wide/Ultra-3 SCSI I/O Controller


cat 1 |awk '/Disk Drive/{gsub(/\(||MB\)/,"");sum+=$NF}END{print sum"MB"}'


#BUG's shell 
cat /etc/passwd | awk -F":" 'BEGIN { total = 0}{ total += $3;}END{print total}'


i want to be 
netstat -anl|awk '{print $3}'| grep -v "^0" |awk 'BEGIN{sum=0}{sum=sum+$0}END{print sum}'


netstat -anl|grep :1935|awk '{print $3}'| grep -v "^0" |awk 'BEGIN{sum=0}{sum=sum+$0}END{print sum}'



程序一：
  彩色显示文件程序，每个字符的颜色都不一样，随机变化，让你看花眼睛^_^：
  
  程序代码：
  
  #! /usr/bin/awk
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
  
  
  
  程序二：
  彩色作图程序,有点像Gnuplot，可以作出圆形，正弦图，抛物线等等。
  
  例如：
     圆：      awk 'BEGIN{while(k<10){print sin(k),cos(k);k=k+0.01}}'   |  awk -f ColorPlot.awk
     正弦线:   awk 'BEGIN{while(k<10){print sin(k),k;k=k+0.01}}'        |  awk -f ColorPlot.awk
     抛物线：  awk 'BEGIN{k=-10;while(k<10){print k^2,k;k=k+0.01}}'     |  awk -f ColorPlot.awk
     直线：    paste <(seq 1 0.01 10)  <(seq 1 0.01 10)                 |  awk -f ColorPlot.awk
  
  
  如果你有想象力的话还可以作出很多意想不到的图形，比如：
  
     圆盘:    awk 'BEGIN{while(k<100){print sin(k),rand()*cos(k);k=k+0.01}}' |awk -f ColorPlot.awk
     花圈:    awk 'BEGIN{srand()
                          while(k++<20000){
                             x=2-3*rand()
                             y=2-4*rand()
                             if(x^2+y^2>0.6&&x^2+y^2<1||x^2+y^2<0.3&&x^2+y^2>0.1)
                                print x,y
                             }
                         }'       | awk -f ColorPlot.awk
  
     菱圈:    awk 'BEGIN{srand()
                          while(k++<20000){
                             x=1-2*rand()
                             y=1-2*rand()
                             if(x+y<=1&&x-y<=1&&-x+y<=1&&-x-y<=1&&x^2+y^2>=1/2)
                                print x,y
                             }
                         }'       | awk -f ColorPlot.awk
  
  爱心型
awk 'BEGIN{while(u<20){print sin(u)*sin(v),rand()*cos(u)*sin(v+u);v=v+0.01;u=u+0.01}}'
awk 'BEGIN{while(u<20){print sin(u)*sin(v),rand()*cos(u)*sin(v+u);v=v+0.01;u=u+0.01}}'|awk -f ColorPlot.awk


  绳结
awk 'BEGIN{while(u<10){print sin(u+v)*sin(v),cos(u+v)*sin(v);v=v+0.01;u=u+0.01}}'|awk -f ColorPlot.awk
  蝴蝶
awk 'BEGIN{while(u<10){print sin(u+v)*sin(v),cos(u)*sin(v);v=v+0.01;u=u+0.01}}'|awk -f ColorPlot.awk
 花瓣
awk 'BEGIN{while(u<10){print sin(u+w)*cos(v)*sin(w+u),sin(u)*sin(v+u)*sin(w);v=v+0.01;u=u+0.01;w=w+0.01}}'|awk -f ColorPlot.awk
 蝙蝠
 awk 'BEGIN{while(u<10){print sin(u+w)*cos(v+w)*sin(w+u),sin(u)*sin(v+u)*sin(w);v=v+0.01;u=u+0.01;w=w+0.01}}'|awk -f ColorPlot.awk
  螺旋
 awk 'BEGIN{while(w<20){print sin(w)*w,cos(w)*w;w=w+0.01}}'|awk -f ColorPlot.awk
  
  
  程序代码：
  
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
