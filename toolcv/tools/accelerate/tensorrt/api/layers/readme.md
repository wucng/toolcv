# 使用说明
- 1.`touch to trt engine` 
	- 1.使用 [layers_trt.py](./layers_trt.py)与[modelTansform.py](../modelTansform.py]) (从头开始搭建)
	- 2.[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

- 2.`touch to onnx` （可以打开 [netron](https://github.com/lutzroeder/Netron) 查看生成的模型结构）
	- 1.`touch.onnx.export()` pytouch api
	- 2.[onnxAPI.py](./onnxAPI.py) (从头开始搭建)
	- 3.[onnxGraphAPI.py](./onnxGraphAPI.py) (从头开始搭建)

- 3.`onnx to trt engine`
	- 1.[modelTansform.py](../modelTansform.py])
	- 2.[onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)

- 4.推荐顺序，
	- 1.`touch.onnx.export()`导出`.onnx`文件
	- 2.使用[onnxGraphAPI.py](./onnxGraphAPI.py)修改某些层结构(如果某层不能转成`trt`)
	- 3.最后转成 `engine`文件（trt文件）

# 示例
```py
1.执行 onnxGraphAPI.py 中的torch2onnx(model, x,'model.onnx') 生成`model.onnx`
2.执行onnx2engine('model.onnx', ModelData=ModelData) 会报错，主要是在 实现 touch模型时 `upsample`操作
3.修改`model.onnx`模型中的`upsample` 执行 OnnxGraphAPI.modeify_model2()，再执行 #2中`onnx2engine`即可
```

- 修改前
（这种方式实现的 upsample 转trt文件时会报错）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722133859913.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)
- 修改后
（这种方式实现的 upsample 转trt文件时不会报错）
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020072210422827.png)
