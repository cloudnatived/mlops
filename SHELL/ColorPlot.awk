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

