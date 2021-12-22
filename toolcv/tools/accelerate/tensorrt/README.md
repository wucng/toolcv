<font size=4>

- https://hub.fastgit.org/shouxieai/tensorRT_cpp  # onnx to trt
- https://hub.fastgit.org/wang-xinyu/tensorrtx    # weight to trt (use trt api)
- https://hub.fastgit.org/NVIDIA/TensorRT

- https://github.com/wucng/Study/tree/master/cuda/tensorrt
- https://developer.nvidia.com/tensorrt
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html
- https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html
- https://github.com/NVIDIA/TensorRT

---
[toc]

# 1、安装
参考官方安装文档[here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)

点击[这里](https://developer.nvidia.com/tensorrt)下载合适版本
```c
// 查看Ubuntu版本
sudo lsb_release -a

// 查看cuda版本
cat /usr/local/cuda/version.txt

// 查看`cudnn`版本
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
// or
cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031172904605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)
`tar文件`提供了更大的灵活性，例如同时安装多个版本的TensorRT。但是，您需要确保已经安装了必要的依赖项，并且必须进行管理 `LD_LIBRARY_PATH` 你自己 有关更多信息，请参见Tar File Installation。

`zip文件`是Windows当前唯一的选项。除了Windows，它不支持任何其他平台。确保您已经安装了必要的依赖项。有关更多信息，请参见Zip文件安装。

## Tar File Installation
```c
// 1、安装依赖项
	- 1、CUDA Toolkit 9.0, 10.0 or 10.1 update 2
	- 2、cuDNN 7.6.3
	- 3、Python 2 or Python 3 (Optional)

// 2、下载TensorRT tar file
TensorRT-6.0.1.5.Ubuntu-16.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz

// 4、解压文件
$ tar zxvf TensorRT-6.0.1.5.Ubuntu-16.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz
$ ls TensorRT-6.0.1.5
bin  data  doc  graphsurgeon  include  lib  python  samples  targets  TensorRT-Release-Notes.pdf  uff

// 5、将TensorRT/lib目录的绝对路径添加到环境变量LD_LIBRARY_PATH中：
// $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/TensorRT-6.0.1.5/lib
// 使用绝对路径
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/content/TensorRT-6.0.1.5/lib

// 6、Install the Python TensorRT wheel file.
$ cd TensorRT-6.0.1.5/python
// If using Python 2.7:
$ sudo pip2 install tensorrt-6.0.1.5-cp27-none-linux_x86_64.whl
// If using Python 3.x:(python3.6)
$ sudo pip3 install tensorrt-6.0.1.5-cp36-none-linux_x86_64.whl

// 7、安装Python UFF wheel文件。 仅当您计划将TensorRT与TensorFlow一起使用时才需要这样做。
$ cd TensorRT-6.0.1.5/uff
// If using Python 2.7:
$ sudo pip2 install uff-0.6.5-py2.py3-none-any.whl
// If using Python 3.x:
$ sudo pip3 install uff-0.6.5-py2.py3-none-any.whl
// 安装完成后 查看安装到的位置
$ which convert-to-uff
/usr/local/bin/convert-to-uff

// 8、安装Python graphsurgeon wheel文件。
$ cd TensorRT-6.0.1.5/graphsurgeon
// If using Python 2.7
$ sudo pip2 install graphsurgeon-0.4.1-py2.py3-none-any.whl
// If using Python 3.x:
$ sudo pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl

// 9、验证安装
- 确保已安装的文件位于正确的目录中。 例如，运行tree -d命令来检查lib，include，data等目录中是否所有受支持的已安装文件都到位。
- 在安装的目录中生成并运行其中一个样本，例如sampleMNIST。 您应该能够编译并执行示例，而无需其他设置。 有关sampleMNSIT的更多信息，请参见TensorRT示例的“ Hello World”。
- Python示例位于samples / python目录中。
```
## Zip File Installation
只是针对`Windows版本`

#  卸载 TensorRT
```c
$ sudo apt-get purge "libnvinfer*"
$ sudo apt-get purge "graphsurgeon-tf"
$ sudo apt-get purge "uff-converter-tf"
$ sudo apt-get autoremove
$ sudo pip3 uninstall tensorrt
$ sudo pip3 uninstall uff
$ sudo pip3 uninstall graphsurgeon
```
# 安装pycuda
```c
pip3 install 'pycuda>=2019.1.1'

pip3 install pycuda -i https://pypi.doubanio.com/simple
```
#  安装 ONNX For Python
```c
// 依赖的包
cmake >= 3.2
protobuf-compiler
libprotoc-dev

// 下面的pip命令将从源代码安装或升级ONNX Python模块，以确保与使用分发编译器构建的TensorRT兼容。 将以下版本替换为TensorRT发行版支持的特定ONNX版本。
pip install --no-binary onnx 'onnx==1.5.0'
```
# 测试
python版
```python
import tensorrt
import uff
import pycuda
import onnx
```
C++版
```python
$ cd TensorRT-XXX/samples/sampleMNIST
$ make clean
$ make
$ cd /TensorRT-XXX/bin（转到bin目录下面，make后的可执行文件在此目录下）
$ ./sample_mnist
```

如果出现以下错误
```c
// 问题
ImportError: libnvinfer.so.6: cannot open shared object file: No such file or directory
// 解决方法
sudo cp TensorRT-6.0.1.5/targets/x86_64-linux-gnu/lib/lib* /usr/lib

// 问题
ImportError: libcublas.so.10: cannot open shared object file: No such file or directory
// 解决方法
sudo cp /usr/local/cuda-10.0/lib64/lib* /usr/lib/
```

---
# 快速上手tensorrt
![](https://img-blog.csdnimg.cn/20191027110053976.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70&ynotemdtimestamp=1572145467628)
# tensorrt 步骤
![](https://img-blog.csdnimg.cn/20191027102610795.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70&ynotemdtimestamp=1572145467628)

# 自定义Plugin
![](https://img-blog.csdnimg.cn/20191027110239881.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70&ynotemdtimestamp=1572145467628)