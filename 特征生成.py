import librosa
import numpy as np
import os
import csv


# 拿取数据直接生成STFT
def spectrogram(file_path):
    audio_data, _ = librosa.load(file_path, sr=None)  # 使用librosa加载file_path里的wav文件，返回值第二个是采样率16000，这里不需要
    spectrogram = librosa.stft(audio_data)
    spectrogram = np.abs(spectrogram)
    return spectrogram

# 拿取数据直接生成梅尔谱
def mel_spectrogram(file_path):
    audio_data, sampleRate = librosa.load(file_path, sr=None)
    print(sampleRate)
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sampleRate)  # 这里的16000是采样率
    return mel_spec

def get_file_path(folder_path):
    # 用于存储.wav文件路径的列表
    wav_files = []
    # 遍历主文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav") and not file.startswith("._"): # 有些wav文件是. 开头的，这些进入librosa会报错，加上这个and not 条件来剔除
                # 构建完整的文件路径并添加到列表中
                wav_path = os.path.join(root, file)
                wav_files.append(wav_path)
    print("已经将"+len(wav_files)+"个数据保存进列表里")

    # 打印列表中的.wav文件路径
    # for wav_file in wav_files:
    #     print(wav_file)
    print(wav_files)
    return wav_files

# 指定包含子文件夹的主文件夹路径
folder_path = 'sentences'

wav_list = get_file_path(folder_path) #这个list可能会非常庞大

for item in wav_list:
    # 获得mel
    spect = mel_spectrogram(item)

    # 提取文件名（不包含文件夹路径和扩展名）
    file_name = os.path.splitext(os.path.basename(item))[0]
    # 保存梅尔谱图数据到CSV文件
    csv_file_path = f'mel/{file_name}.csv'

    # 将梅尔谱图数据写入CSV文件
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in spect:
            csv_writer.writerow(row)

    print(f"梅尔谱图数据已成功写入到 {csv_file_path}")


