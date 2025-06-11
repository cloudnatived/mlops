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
