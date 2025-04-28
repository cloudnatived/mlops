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


