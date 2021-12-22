# // 有问题 未实现
能正常编译 但调用时报错

# alexnet

AlexNet model architecture from the "One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

For the details, you can refer to [pytorchx/alexnet](https://github.com/wang-xinyu/pytorchx/tree/master/alexnet)

This alexnet is just several `conv-relu-pool` blocks followed by several `fc-relu`, nothing special. All layers can be implemented by tensorrt api, including `addConvolution`, `addActivation`, `addPooling`, `addFullyConnected`.

```
// 1. generate alexnet.wts from [pytorchx/alexnet](https://github.com/wang-xinyu/pytorchx/tree/master/alexnet)

// 2. put alexnet.wts into tensorrtx/alexnet

// 3. build and run

cd tensorrtx/alexnet

mkdir build

cd build

cmake ..

make

sudo ./alexnet -s   // serialize model to plan file i.e. 'alexnet.engine'

sudo ./alexnet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/alexnet
```

# 结果对比

- float32
```py
# python wts2trt
534
Output:
[-0.7273843   0.7084095  -1.8970122  -2.2323935  -1.0149302  -0.04174021
 -3.0588741  -0.3024685  -2.070097    0.08364692]
[-0.16299678 -2.6806471  -0.9106289  -0.92323166 -0.6023281  -1.061207
 -1.1566269  -1.9356979  -0.7136966   0.85299534]

# python onnx2trt
534
Output:
[-0.7273843   0.7084095  -1.8970122  -2.2323935  -1.0149302  -0.04174021
 -3.0588741  -0.3024685  -2.070097    0.08364692]
[-0.16299678 -2.6806471  -0.9106289  -0.92323166 -0.6023281  -1.061207
 -1.1566269  -1.9356979  -0.7136966   0.85299534]

# cpp wts2trt
534
-0.727384, 0.708409, -1.89701, -2.23239, -1.01493, -0.0417402, -3.05887, -0.302469, -2.0701, 0.0836469,
-0.162997, -2.68065, -0.910629, -0.923232, -0.602328, -1.06121, -1.15663, -1.9357, -0.713697, 0.852995,


# cpp onnx2trt
534
-0.727384, 0.708409, -1.89701, -2.23239, -1.01493, -0.0417402, -3.05887, -0.302469, -2.0701, 0.0836469,
-0.162997, -2.68065, -0.910629, -0.923232, -0.602328, -1.06121, -1.15663, -1.9357, -0.713697, 0.852995,

```

- float16

```py
# python wts2trt (结果不对)
783
Output:
[-1.2333742   1.2734455  -0.8746761  -1.0386852  -0.30626684 -0.12559775
 -1.5416714   0.59901226 -1.333971    0.81947255]
[ 0.10147394 -1.9814379  -0.09485485 -0.9673006   0.04698183 -0.570666
 -1.897552   -1.1840495  -0.4934864   1.3988563 ]


# python onnx2trt (结果不对)
783
Output:
[-1.2333742   1.2734455  -0.8746761  -1.0386852  -0.30626684 -0.12559775
 -1.5416714   0.59901226 -1.333971    0.81947255]
[ 0.10147394 -1.9814379  -0.09485485 -0.9673006   0.04698183 -0.570666
 -1.897552   -1.1840495  -0.4934864   1.3988563 ]

# cpp wts2trt
534
-0.719727, 0.706543, -1.89844, -2.21484, -1.00879, -0.0311127, -3.07422, -0.310791, -2.07227, 0.0547485,
-0.162842, -2.68359, -0.921875, -0.92334, -0.614258, -1.07324, -1.13672, -1.93652, -0.702637, 0.854004,


# cpp onnx2trt
534
-0.719727, 0.706543, -1.89844, -2.21484, -1.00879, -0.0311127, -3.07422, -0.310791, -2.07227, 0.0547485,
-0.162842, -2.68359, -0.921875, -0.92334, -0.614258, -1.07324, -1.13672, -1.93652, -0.702637, 0.854004,

```


