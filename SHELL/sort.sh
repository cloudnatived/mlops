



Shell脚本学习：快速理解正则表达式之grep篇
[日期：2011-07-24] 	来源：Linux社区  作者：deansrk

shell脚本是Linux的核心之一，而正则表达式是shell脚本的核心之一，理解正则表达式可以快速匹配需要查找的内容，对以后的shell脚本编程打下一个坚实的基础。

接触正则表达式一般从grep命令开始，例如：

grep "^#[:space:]*" /etc/inittab

这个命令中“^#[:space:]*”就是正则表达式的内容，它的意思是“以#开头后跟任意个空格”,结合grep命令，就是显示/etc/inittab文件里“以#开头后跟任意个空格”的内容。

grep命令的作用是查找匹配的内容并打印出来

grep [option]  正则表达式 要查抄的文件

-i   不区分大小写

-color    以高亮显示匹配的文本内容

-E  使用扩展的元字符

-v   取反

那么正则表达式分有哪些？

——————————————————————

     在shell里，正则表达式分为，标准元字符、扩展元字符

     标准元字符： ^  $  .  *  []  [x-y]  [^]  \  \<  \>  \(...)\   x\{m\n}

     扩展元字符： ^  $  .  *  []  [^]   +  ?   a|b   ()          

     相关资料：http://www.linuxidc.com/Linux/2011-07/39153p2.htm

      #可在上面的链接里查到各自的意思

      另外一个重要的知识：posix方括号字符集       

       [:alnum:]            [:lower:]          [:xdigit:]

       [:alpha:]             [:print:]            [:blank:]

       [:blank:]             [:punct:]

       [:cntrl:]               [:space:]

       [:graph:]             [::upper:]

       上面的例子grep "^#[:space:]*" /etc/inittab 里[:space:]代表空白字符，posix字符集和正则表达式配合使用可以产生很强大的功能，许多时候我们都可以借助它来实现自己的目的。

       例如：显示/boot/grub/grub.conf文件中以一个或多个空白字符开头的行

       grep "^[[:space:]]\{1,\}" /boot/grub/grub.conf

       这个命令里注意 ^和[]的使用，通常^[]用来匹配开头是某个字符，^[[:space:]]因为要求匹配的是开头是空白字符，所以用[[:space:]]而不是 [:space:],如果错误的写成^[:space:]那么就锚定[]里的内容，现在知道[]的作用了吧：锚定某个字符，多个代表多个可能 

   下面几个难度稍高一点的练习来学习理解正则表达式

   1.显示/etc/inittab文件中以一个数字开头并以一个与开头数字相同的数字结尾的行；
   2.ifconfig命令可以显示当前主机的IP地址相关的信息等，如果使用grep等文本处理命令取出本机的各IP地址，要求不包括127.0.0.1；
   3.显示/etc/sysconfig/network-scripts/ifcfg-eth0文件中的包含了类似IP地址点分十进制数字格式的行； 

##答案：

  1.  grep "^\([0-9]\).*\1$" /etc/inittab

  2.  ifcofig | grep "inet addr" | grep -v '127.0.0.1' | cut -d： -f2 | cut -d "" f1

  3.  grep -E "([0-9]{1,3}\.){3}\.[0-9]{1,3}" /etc/sysconfig/network-scripts/ifcfg-eth0

       grep "[0-9]\{1,3\}\.\"{3\}\.[0-9]\{1,3\} /etc/sysconfig/network-scripts/ifcfg-eth0 "

##解析

   1. ^[0-9]锚定开头的数字 使用\(...\)   \1   将^[0-9]传递给\1 $用来锚定结尾

   2.  首先grep提出去含有inet addr的内容,-v取出‘127.0.0.1’使用cut命令 -d 来截取第一个字段

   3.  使用-E解法 ([0-9])\{1,3}    [0-9]的数字至少出现1次，至多出现3次   \.  转义 . 符号 {3} 显示三次前面()的内容\.[0-9]{1,3}匹配后面的3位数字                                    

        ##grep解法可以自己琢磨下




正则表达式 (grep)     grep (global search regular expression(RE) and print out the line,全面搜索正则表达式并把行打印出来)是一种强大的文本搜索工具，它能使用正则表达式搜索文本，并把匹配的行打印出来。搜索的结果被送到屏幕，不影响原文件内容。Unix的grep家族包括grep、 egrep和fgrep。egrep和fgrep的命令只跟grep有很小不同。egrep是grep的扩展，支持更多的re元字符，一:语法 grep -aceinv ‘字符串' filename 参数说明： -a 当对binary文件搜索时使用 -c 计算次数 -e 两个表达式连到一起 -i 忽略大小写的不同 -n 输出行号 -v 反向选择   字符串     ^word  搜索行首为(word)的行     word$  搜索行尾为(word)的行     .     代表任意一个字符，一定是一个任意字符     \       将特殊符号的特殊意义去除     *      重复零个或多个的前一个 RE 字符     [ ]     代表一个待搜索的字符     [ - ]     中的减号 -代表两个字符之间的所有连续字符     [^abc ]  代表不包括abc 如grep -n [^A-Z]不搜索大写     \{n,m\}  连续 n 到 m 个的(前一个 RE 字符)     \{n\}    连续 n 个的前一个 RE 字符     \{n,\}   连续 n 个以上的前一个 RE 字符     \(..\)   一个字符单位，如'\(love\)'，love被标记为1。       egrep 是 grep CE     +    重复零个或多个的前一个 RE 字符     ?    零个或一个前一个 RE 字符     |    用或( or )的方式找出数个字符串     ( )   一个字符单位二:在线验证       
shell grep 正则
三:常用正则表达式 1.常用的正则表达式     [\u4e00-\u9fa5]      //匹配中文字符     [^\x00-\xff]            //匹配双字节字符(包括汉字在内)     \n\s*\r                     //匹配空白行的正则表达式     ^\s*|\s*$                //匹配首尾空白字符     [a-zA-z]+://[^\s]*    //匹配网址URL     ^[a-zA-Z][a-zA-Z0-9_]{4,15}$  //匹配帐号是否合法(字母开头，允许5-16字节，允许字母数字下划线)     \d{3}-\d{8}|\d{4}-\d{7}        //匹配国内电话号码     [1-9][0-9]{4,}                        //匹配腾讯QQ号     [1-9]\d{5}(?!\d)                     //匹配中国邮政编码     \d{15}|\d{18}                        //匹配身份证     \d+\.\d+\.\d+\.\d+                   //匹配ip地址     <(\S*?)[^>]*>.*?</\1>|<.*? />             //匹配HTML标记的正则表达式     \w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*   //匹配Email地址 2.匹配特定数字     ^[1-9]\d*$    　 //匹配正整数     ^-[1-9]\d*$ 　  //匹配负整数     ^-?[1-9]\d*$　  //匹配整数     ^[1-9]\d*|0$　  //匹配非负整数（正整数 + 0）     ^-[1-9]\d*|0$　 //匹配非正整数（负整数 + 0）     ^[1-9]\d*\.\d*|0\.\d*[1-9]\d*$　　 //匹配正浮点数     ^-([1-9]\d*\.\d*|0\.\d*[1-9]\d*)$　 //匹配负浮点数     ^-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$　     //匹配浮点数     ^[1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0$　　     //匹配非负浮点数+ 0      ^(-([1-9]\d*\.\d*|0\.\d*[1-9]\d*))|0?\.0+|0$　　//匹配非正浮点数+0
3.匹配特定字符串     ^[A-Za-z]+$　  //匹配由26个英文字母组成的字符串     ^[A-Z]+$　　      //匹配由26个英文字母的大写组成的字符串     ^[a-z]+$　　       //匹配由26个英文字母的小写组成的字符串     ^[A-Za-z0-9]+$　//匹配由数字和26个英文字母组成的字符串     ^\w+$　　             //匹配由数字、26个英文字母或者下划线组成的字符串

本文出自 “” 博客，请务必保留此出处http://chenxy.blog.51cto.com/729966/178738
'







sort -k1n -k2n







正则表达式与grep ,在指南的第7和8章

shell 阅读笔记－正则表达式与grep

一  正则表达式
基本元字符集及其含义
    ^     只只匹配行首
    $     只只匹配行尾
    *     只一个单字符后紧跟*，匹配0个或多个此单字符
    [ ]   只匹配[ ]内字符。可以是一个单字符，也可以是字符序列。可以使用-
            表示[ ]内字符序列范围，如用[ 1 - 5 ]代替[ 1 2 3 4 5 ]
    \     只用来屏蔽一个元字符的特殊含义。因为有时在shell中一些元字符有
           特殊含义。\可以使其失去应有意义
    .     只匹配任意单字符
  pattern\      只用来匹配前面pattern出现次数。n为次数
  pattern\m    只含义同上，但次数最少为n
  pattern\    只含义同上，但pattern出现次数在n与m之间

二  grep 的用法
 1 双引号引用
   在grep命令中输入字符串参数时，最好将其用双引号括起来
 2 grep选项
   常用的g r e p选项有：
   -c 只输出匹配行的计数。
   -i 不区分大小写（只适用于单字符）。
   -h 查询多文件时不显示文件名。
   -l 查询多文件时只输出包含匹配字符的文件名。
   -n 显示匹配行及行号。
   -s 不显示不存在或无匹配文本的错误信息。
   -v 显示不包含匹配文本的所有行。
 3  精确匹配
    使用grep抽取精确匹配的一种更有效方式是在抽取字符串后加\>。假定现在精确抽取48，则为"48\>"
三 grep和正则表达式
     使用正则表达式时最好用单引号括起来，这样可以防止grep中使用的专有模式与一些shell命令的特殊方式相混淆。
  1 模式范围
    假定要抽取代码为484和483的城市位置，上一章中讲到可以使用[ ]来指定字符串范围，这里用48开始，
    以3或4结尾，这样抽出484或483。grep '48[34]' data.f
  2 不匹配行首
     如果要抽出记录，使其行首不是48，可以在方括号中使用^记号，表明查询在行首开始。
     grep '^[^48]' data.f
  3 匹配任意字符
   如果抽取以L开头，以D结尾的所有代码，可使用下述方法，因为已知代码长度为5个字符：
    grep 'L...D' data.f
  4 日期查询
    一个常用的查询模式是日期查询。先查询所有以5开始以1 9 9 6或1 9 9 8结尾的所有记录。使用模式5..199[6,8].这意味着第一个字符为

    5，后跟两个点，接着是199，剩余两个数字是6或8。
     grep '5..199[6,8]' data.f
    查询包含1998的所有记录的另外一种方法是使用表达式[0-9]\[8]，含义是任意数字重复3次，后跟数字8，虽然这个方法不像上一个方

   法那么精确，但也有一定作用。
  5 使用grep匹配“与”或者“或”模式
    grep命令加- E参数，这一扩展允许使用扩展模式匹配。例如，要抽取城市代码为2 1 9或2 1 6，方法如下：
    grep -E '219|216' data.f
  6 空行
   结合使用^和$可查询空行。使用- n参数显示实际行数：
    grep '^$' data.f
  7 grep 与类名的使用
   类等价的正则表达式类等价的正则表达式
  [[:upper:]]   [A-Z]               [[:alnum:]]  [0-9a-zA-Z]
  [[:lower:]]   [a-z]               [[:space:]] 空格或t a b键
  [[:digit:]]   [0-9]                [[:alpha:]] [a-zA-Z]





sort(分类)用法
lhccie
sort(分类)用法

这里的内容有很多和wingger的内容一样，只是有些地方不一样，如果有上面不正确的地方希望大家指点。
wingger内容的原始链接[url]http://bbs.chinaunix.net/viewthread.php?tid=457730###[/url]

1.sort(分类)用法

sort命令选项很长，下面仅介绍各种选项。
sort命令的一般格式为：

sort -cmu -o output_file [other options] +pos1 +pos2 input_files

下面简要介绍一下sort的参数：

-c 测试文件是否已经分类。
-m 合并两个分类文件。
-u 删除所有复制行。
-o 存储s o r t结果的输出文件名。


其他选项有：
-b 使用域进行分类时，忽略第一个空格。
-n 指定分类是域上的数字分类。
-t 域分隔符；用非空格或t a b键分隔域。
-r 对分类次序或比较求逆。
+n n为域号。使用此域号开始分类。
n n为域号。在分类比较时忽略此域，一般与+n一起使用。
post1 传递到m，n。m为域号，n为开始分类字符数；例如4，6意即以第5域分类，从第7个字符开始。

保存输出
- o选项保存分类结果，然而也可以使用重定向方法保存。下面例子保存结果到results.out：
$sort video >results.out

2.例子说明
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
下面是文件video的清单，包含了上个季度家电商场的租金情况。各域为：
(1)名称，(2)供货区代码，(3)本季度租金,(4)本年租金。

域分隔符为冒号,为此对此例需使用‘-t’选项。文件如下：
[root@Linuxsvr lab]# cat video
Boys in Company C       :HK     :192    :2192
Alien                   :HK     :119    :1982
The Hill                :KL     :63     :2972
Aliens                  :HK     :532    :4892
Star Wars            :HK     :301    :4102
A Few Good Men :KL     :445    :5851
Toy Story             :HK     :239    :3972

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

3.启动方式
缺省情况下，sort认为“一个空格”或“一系列空格”为“分隔符”。要加入其他方式分隔，使用-t(-t+分隔符)选项，如：
[root@Linuxsvr lab]# sort -t: video
A Few Good Men          :KL     :445    :5851
Alien                   :HK     :119    :1982
Aliens                  :HK     :532    :4892
Boys in Company C       :HK     :192    :2192
Star Wars               :HK     :301    :4102
The Hill                :KL     :63     :2972
Toy Story               :HK     :239    :3972

以“:”为分隔符，按照第一列排序

4.
先查看是否为域分隔设置了- t选项，如果设置了- t选项，则使用分隔符将记录分隔成 域0、域1、域2、域3等等；
如果未设置- t选项，用空格代替。
缺省时sort以每行第一个字符将整个行排序，也可以指定域号，这样就会按照指定的域优先进行排序，
如果指定的域有重复，会参考下一个域。

sort对域的参照方式：
关于sort的一个重要事实是它参照第一个域作为域0，域1是第二个域，等等。sort也可以使用整行作为分类依据。

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
第一个域             第二个域 第三个域  第四个域
域0                        域1        域2        域3
Boys in Company C       :HK     :192    :2192
Alien                   :HK     :119    :1982
The Hill                :KL     :63     :2972
Aliens                  :HK     :532    :4892
Star Wars               :HK     :301    :4102
A Few Good Men          :KL     :445    :5851
Toy Story               :HK     :239    :3972

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

文件是否已分类
怎样分辨文件是否已分类？如果只有30行，看看就知道了，但如果是400行呢，使用sort -c查看sort文件是否按某种顺序分类。
[root@Linuxsvr lab]# sort -c video
sort: video:2: disorder: Alien                  :HK     :119    :1982
结果显示未分类，现在将video分类，并存为2.video
[root@Linuxsvr lab]# sort -t: video >2.video
[root@Linuxsvr lab]# sort -c 2.video
[root@Linuxsvr lab]#
没有任何错误提示，返回提示符表明已分类。然而如果测试成功，返回一个信息行会更好。

5.
基本sort

最基本的sort方式为sort filename，按第一域进行分类（分类键0）。实际上读文件时sort操作将行中各域进行比较，
这里返回基于第一域sort的结果
[root@Linuxsvr lab]# sort -t: video

A Few Good Men          :KL     :445    :5851
Alien                   :HK     :119    :1982
Aliens                  :HK     :532    :4892
Boys in Company C       :HK     :192    :2192
Star Wars               :HK     :301    :4102
The Hill                :KL     :63     :2972
Toy Story               :HK     :239    :3972

sort分类求逆
如果要逆向sort结果，使用- r选项。在通读大的注册文件时，使用逆向sort很方便。下面是按域0分类的逆向结果。
[root@Linuxsvr lab]# sort -t: -r video
Toy Story               :HK     :239    :3972
The Hill                :KL     :63     :2972
Star Wars               :HK     :301    :4102
Boys in Company C       :HK     :192    :2192
Aliens                  :HK     :532    :4892
Alien                   :HK     :119    :1982
A Few Good Men          :KL     :445    :5851

按指定域分类
有时需要只按第2域（分类键1）分类。这里为重排 报文中“供应区代码”，使用t 1，意义为按分类键1分类。
下面的例子中，所有供应区代码按分类键1分类；
注意分类键2和3对应各域也被分类。因为第2域有重复，sort会再重复的情况下优先考虑下一个域的顺序。而且是按照第一个字符分类，并不是

按照整个数值大小分类63小于445，却被排到后面，因为第一个字符是6，大于4。

[root@Linuxsvr lab]# sort -t: +1 video

Alien                   :HK     :119    :1982
Boys in Company C       :HK     :192    :2192
Toy Story               :HK     :239    :3972
Star Wars               :HK     :301    :4102
Aliens                  :HK     :532    :4892
A Few Good Men          :KL     :445    :5851
The Hill                :KL     :63     :2972

数值域分类
依此类推，要按第三域(第二分类键)分类，使用t 2。但是因为这是数值域，即为数值分类，可以使用- n选项。下面例子为按季度租金分类命

令及结果：
[root@Linuxsvr lab]# sort -t: +2n video

The Hill                :KL     :63     :2972
Alien                   :HK     :119    :1982
Boys in Company C       :HK     :192    :2192
Toy Story               :HK     :239    :3972
Star Wars               :HK     :301    :4102
A Few Good Men          :KL     :445    :5851
Aliens                  :HK     :532    :4892
使用-n选项是按照数值大小进行排列的，不使用－n选项是按照数字位排列，先看最左边第一位大小，如果第一位相同再看第二位大小

如果不指定n,如下
[root@Linuxsvr lab]# sort -t: +2 video

Alien                   :HK     :119    :1982
Boys in Company C       :HK     :192    :2192
Toy Story               :HK     :239    :3972
Star Wars               :HK     :301    :4102
A Few Good Men          :KL     :445    :5851
Aliens                  :HK     :532    :4892
The Hill                :KL     :63     :2972

数值域倒序：

[root@Linuxsvr lab]# sort -t: +2nr video
Aliens                  :HK     :532    :4892
A Few Good Men          :KL     :445    :5851
Star Wars               :HK     :301    :4102
Toy Story               :HK     :239    :3972
Boys in Company C       :HK     :192    :2192
Alien                   :HK     :119    :1982
The Hill                :KL     :63     :2972

唯一性分类
有时，原文件中有重复行，这时可以使用- u选项进行唯一性（不重复）分类以去除重复行，下例中A l i e n有相同的两行。带重复行的文件

如下，其中A l i e n插入了两次：
[root@Linuxsvr lab]# cat video
Boys in Company C       :HK     :192    :2192
Alien                   :HK     :119    :1982
The Hill                :KL     :63     :2972
Aliens                  :HK     :532    :4892
Star Wars               :HK     :301    :4102
A Few Good Men          :KL     :445    :5851
Toy Story               :HK     :239    :3972
Aliens                  :HK     :532    :4892

使用- u选项去除重复行，不必加其他选项， sort会自动处理。
[root@Linuxsvr lab]# sort -u video
A Few Good Men          :KL     :445    :5851
Alien                   :HK     :119    :1982
Aliens                  :HK     :532    :4892
Boys in Company C       :HK     :192    :2192
Star Wars               :HK     :301    :4102
The Hill                :KL     :63     :2972
Toy Story               :HK     :239    :3972
[root@Linuxsvr lab]# sort video
A Few Good Men          :KL     :445    :5851
Alien                   :HK     :119    :1982
Aliens                  :HK     :532    :4892
Aliens                  :HK     :532    :4892
Boys in Company C       :HK     :192    :2192
Star Wars               :HK     :301    :4102
The Hill                :KL     :63     :2972
Toy Story               :HK     :239    :3972

使用k的其他sort方法
sort还有另外一些方法指定分类键。可以指定k选项，第1域（分类键）以1开始。不要与前面相混淆。其他选项也可以使用k，主要用于指定分

类域开始的字符数目。格式：
-k  keydef
The keydef argument is a restricted sort key field  definition. The format of this definition is:

[root@Linuxsvr lab]# sort -t: -k[field_start[type][,field_end[type]]] video              

[root@Linuxsvr lab]# sort -t: -k2,2 -k1,1 video
Alien                   :HK     :119    :1982
Aliens                  :HK     :532    :4892
Aliens                  :HK     :532    :4892
Boys in Company C       :HK     :192    :2192
Star Wars               :HK     :301    :4102
Toy Story               :HK     :239    :3972
A Few Good Men          :KL     :445    :5851
The Hill                :KL     :63     :2972

如果不指定结束域，分类将会按照后面的域以次排序。如果上面的例子不指定-k2,2后面结束域，结果如下：
[root@Linuxsvr lab]# sort -t: -k2 -k1,1 video
Alien                   :HK     :119    :1982
Boys in Company C       :HK     :192    :2192
Toy Story               :HK     :239    :3972
Star Wars               :HK     :301    :4102
Aliens                  :HK     :532    :4892
Aliens                  :HK     :532    :4892
A Few Good Men          :KL     :445    :5851
The Hill                :KL     :63     :2972
上面的例子不会再以第一域排序，而是按照第二域排序，如果第二域有重复，优先考虑第三域，如果再有重复，考虑第四域，而不是第一域。


用k做分类键排序
可以指定分类键次序。再全部将结果反向排序，方法如下：
[root@Linuxsvr lab]# sort -t: -k2,2 -k1,1 -r video
The Hill                :KL     :63     :2972
A Few Good Men          :KL     :445    :5851
Toy Story               :HK     :239    :3972
Star Wars               :HK     :301    :4102
Boys in Company C       :HK     :192    :2192
Aliens                  :HK     :532    :4892
Aliens                  :HK     :532    :4892
Alien                   :HK     :119    :1982
[root@Linuxsvr lab]# sort -t: -k2,2 -k1,1 video
Alien                   :HK     :119    :1982
Aliens                  :HK     :532    :4892
Aliens                  :HK     :532    :4892
Boys in Company C       :HK     :192    :2192
Star Wars               :HK     :301    :4102
Toy Story               :HK     :239    :3972
A Few Good Men          :KL     :445    :5851
The Hill                :KL     :63     :2972

下面的例子把Aliens改为Bliens，先对第三域反向排序，重复的地方再按照第一域正向排。
[root@Linuxsvr lab]# sort -t: +2nr -k1,1 video
Aliens                  :HK     :532    :4892
Bliens                  :HK     :532    :4892
A Few Good Men          :KL     :445    :5851
Star Wars               :HK     :301    :4102
Toy Story               :HK     :239    :3972
Boys in Company C       :HK     :192    :2192
Alien                   :HK     :119    :1982
The Hill                :KL     :63     :2972
下面例子是先对第三域 正向排，重复的地方再按照第一域反向排
[root@Linuxsvr lab]# sort -t: +2n -k1,1 -r video
The Hill                :KL     :63     :2972
Alien                   :HK     :119    :1982
Boys in Company C       :HK     :192    :2192
Toy Story               :HK     :239    :3972
Star Wars               :HK     :301    :4102
A Few Good Men          :KL     :445    :5851
Bliens                  :HK     :532    :4892
Aliens                  :HK     :532    :4892

scode
不错哦。。。顶一下

看图说话
恩，学习一下^_^

-_-
[quote]
`-g'
`--general-numeric-sort'
     Sort numerically, using the standard C function `strtod' to convert
     a prefix of each line to a double-precision floating point number.
     This allows floating point numbers to be specified in scientific
     notation, like `1.0e-34' and `10e100'.  The `LC_NUMERIC' locale
     determines the decimal-point character.  Do not report overflow,
     underflow, or conversion errors.  Use the following collating
     sequence:

        * Lines that do not start with numbers (all considered to be
          equal).

        * NaNs ("Not a Number" values, in IEEE floating point
          arithmetic) in a consistent but machine-dependent order.

        * Minus infinity.

* Finite numbers in ascending numeric order (with -0 and +0
          equal).

        * Plus infinity.

     Use this option only if there is no alternative; it is much slower
     than `--numeric-sort' (`-n') and it can lose information when
     converting to floating point.


`-n'
`--numeric-sort'
     Sort numerically: the number begins each line; specifically, it
     consists of optional whitespace, an optional `-' sign, and zero or
     more digits possibly separated by thousands separators, optionally
     followed by a decimal-point character and zero or more digits.
     The `LC_NUMERIC' locale specifies the decimal-point character and
     thousands separator.

     Numeric sort uses what might be considered an unconventional method
     to compare strings representing floating point numbers.  Rather
     than first converting each string to the C `double' type and then
     comparing those values, `sort' aligns the decimal-point characters
     in the two strings and compares the strings a character at a time.
     One benefit of using this approach is its speed.  In practice
     this is much more efficient than performing the two corresponding
     string-to-double (or even string-to-integer) conversions and then
     comparing doubles.  In addition, there is no corresponding loss of
     precision.  Converting each string to `double' before comparison
     would limit precision to about 16 digits on most systems.

     Neither a leading `+' nor exponential notation is recognized.  To
     compare such strings numerically, use the `--general-numeric-sort'
     (`-g') option.

[/quote]





sort [options] [files]

常见参数：
-b, --ignore-leading-blanks
忽略每行前面开始处的空格和tab字符
-c, --check
检查文件是否已经排序，如果输入文件排序不正确，就返回一个非零值。
-d, --dictionary-order
按字典顺序，即对英文字母、数字及空格字符排序
-f, --ignore-case
排序时，忽略大小写的区别，全部作为大写字母进行
-g, --general-numeric-sort
按常规数字顺序排序
--help
帮助信息
-i, --ignore-nonprinting
忽略不可打印的字符（即指非八进制040~176之间的ASCII字符）排序
-k n[,m] , --key=n[,m]
指定一个或几个字段作为排序关键字，字段位置从n开始，到m为止（包括n，不包括m）。如不指定m，则关键字为从n到行尾。字段和字符的位置从0开始，第一列为1。
-n
按算术大小排序
-ofile, --output=file
将排序结果保存成指定的文件，而非输出到屏幕
-m, --merge
合并几个已经排序的文件
-r, --reverse
反向排序
-s, --stable
关闭最后重排的动作，实现稳定排序
-t:, --field-separator=:
指定列分隔符，默认是tab
-u, --unique
对排序后认为相同的行只留其中一行
-z, --zero-terminated
结束行为0字符，而非新行（\n）字符
--version
显示版本信息
-M, --month-sort
将前3个字母（不含空格、忽略大小写）按照月份缩写进行排序，非月份缩写的行则排在最后，如JAN < FEB
-Ssize, --buffer-size=size
设置多大的缓存，默认1024K，可使用M指定
-T tempdir, --temporary-directory=dir
存放临时文件的目录
 
最有趣的应该是-t参数了，举个例子：
# more test.txt
time0 | userA | a | 1
time3 | userD | d | 3
time1 | userC | c | 4
time2 | userB | b | 2
如果我要按第一列来排序，后面的对应关系还不能错，就要用到-t参数了：sort的-t选项可以实现cut的-d功能，再利用+m -n参数可以实现cut的-f的功能，只是，sort的这个+m -n是从0开始计数的。+m -n是指从第m个字段开始，到第n个字段排序，其中包含第m个但不包含第n个。
比如：以下就定义 “|” 为一个字段。
# sort -t "|"  +0 -1 test.txt
time0 | userA | a | 1
time1 | userC | c | 4
time2 | userB | b | 2
time3 | userD | d | 3
这个 +0 -1 就表示第一列。
 
那么要按第二列或者第三列排序呢？
# sort -t "|"  +1 -2 test.txt
time0 | userA | a | 1
time2 | userB | b | 2
time1 | userC | c | 4
time3 | userD | d | 3

# sort -t "|"  +2 -3 test.txt
time0 | userA | a | 1
time2 | userB | b | 2
time1 | userC | c | 4
time3 | userD | d | 3

# sort -t "|"  +3 -4 test.txt
time0 | userA | a | 1
time2 | userB | b | 2
time3 | userD | d | 3
time1 | userC | c | 4
 
另一个例子：
192.168.19.11
192.168.19.12
192.168.19.8
192.168.19.9
192.168.21.11
192.168.21.12
192.168.21.9
192.168.21.10
192.168.21.5
192.168.19.10

sort -n -t "." +2 -3 +3 -4 sort.txt
192.168.19.8
192.168.19.9
192.168.19.10
192.168.19.11
192.168.19.12
192.168.21.5
192.168.21.9
192.168.21.10
192.168.21.11
192.168.21.12

-k的例子：按不同列排序
[root@test1 tmp]# more 1.txt
c  2  F
a  3  H
b  1  G
[root@test1 tmp]# sort -k1 1.txt
a  3  H
b  1  G
c  2  F
[root@test1 tmp]# sort -k2 1.txt
b  1  G
c  2  F
a  3  H
[root@test1 tmp]# sort -k3 1.txt
c  2  F
b  1  G
a  3  H

 

uniq [options] [files]

 

常用参数：

-c 在输出行前面加上每行在输入文件中出现的次数。


-d 仅显示重复行。


-f Fields 忽略由 Fields 变量指定的字段数目。 如果 Fields 变量的值超过输入行中的字段数目, uniq 命令用空字符串进行比较。 这个标志和 -Fields 标志是等价的。


-u 仅显示不重复的行。


-s Characters 忽略由 Characters 变量指定的字符的数目。 如果 Characters 变量的值超过输入行中的字符的数目, uniq 用空字符串进行比较。 如果同时指定 -f 和 -s 标志, uniq 命令忽略由 -s Characters 标志指定的字符的数目，而从由 -f Fields 标志指定的字段后开始。 这个标志和 +Characters 标志是等价的。


-Fields 忽略由 Fields 变量指定的字段数目。 这个标志和 -f Fields 标志是等价的。
+Characters 忽略由 Characters 变量指定的字符的数目。 如果同时指定 - Fields 和 +Characters 标志, uniq 命令忽略由 +Characters 标志指定的字符数目，并从由 -Fields 标志指定的字段后开始。 这个标志和 -s Characters 标志是等价的。

cut

cut用来从标准输入或文本文件中剪切列或域。剪切文本可以将之粘贴到一个文本文件。

cut  options  file1  file2

参数：

-c   list  指定剪切字符数

-f   field 指定剪切域数

-d         指定与空格和tab键不同的域分隔符

-c 1，5-7  剪切第1个字符，然后是第5到第7个字符

-c1-50     剪切前50个字符

-f         格式与-c相同

-f 1，5    剪切第1域和第5域

-f 1，7-9  剪切第1域，第10域到第12域

eg：

cut -f2- -d "1"  打印以1为域分隔符的第2到最后一域
