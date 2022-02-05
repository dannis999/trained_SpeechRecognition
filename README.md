此项目用于备份一个完整的中文语音识别环境，包括环境配置和预训练模型，以方便直接使用

- 代码来源：https://github.com/nl8590687/ASRT_SpeechRecognition

环境配置：

```
virtualenv --python=%appdata%\..\Local\Programs\Python\Python37\python.exe venv37tf2
cd venv37tf2
Scripts\activate
# cd ...
pip install -r requirements.txt
```

简答测试了一下，感觉拼音准确率还行，但是文字不太行，还需要找别的拼音转文字的项目

