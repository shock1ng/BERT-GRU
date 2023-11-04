# BERT-GRU
BERT承接文本的embedding，GRU承接频谱图，进行多模态情感识别
1.
首先要运行“特征生成”，需要安装好librosa
我在特征生成里面保存了两个函数用于产生不同的特征，分别是spectrogram函数用于生成STFT，mel_spectrogram用于生成mel谱图
我默认使用的是后者，可以看需求使用
在
# 保存梅尔谱图数据到CSV文件
    csv_file_path = f'mel/{file_name}.csv'
这里mel是要保存的进的文件夹名称，大括号内的是每个语音不同的id，不用管它，有需要可以改mel，这里的mel是下一个data_pp的启动文件夹

2.
运行data_pp 需要安装好sklearn和pickle
name那里是你上一步保存的所有csv的文件夹
IEMOCAP_10039.csv由于太大上传不了，这里是一个数据集的小集成
一般来说只要改name的赋值就可以了，直接运行，大概半小时内可以跑出来一个pickle，pickle的名字是name.pickle

3.
然后直接运行train就可以了
最后的结果会生成在一个txt里
