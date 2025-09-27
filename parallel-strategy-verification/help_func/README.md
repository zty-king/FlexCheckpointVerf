# 两个辅助函数

## compare_checkpoints

用来帮助检验转换前的ckpt和转换回来的ckpt二者的md5是否完全一致，若一致则会输出：MD5匹配通过！

## plot_loss

用来画loss函数曲线

## coculate_loss_with_md5

用来计算loss的差值，若md5不相等

## 使用方法
将两个py文件放在与bash脚本的同级目录即可。bash脚本与示例脚本一样，放在`paddlenlp/llm/`下