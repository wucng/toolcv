```py
import numpy as np

def softmax(x,better=True):
	# 在指数上减去输入中的最大值，进行溢出抑制
	if better:x -= x.max(axis=1,keepdims=True)
	t = np.exp(x)
	return t/np.sum(t,axis=1,keepdims=True)
```