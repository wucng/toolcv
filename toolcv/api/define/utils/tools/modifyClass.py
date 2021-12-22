import torch


class T:
    b = 2

    def test01(self):
        print('test01')

    def test02(self, a):
        print("test02", a)

    def test03(self, a):
        print("test03", a, self.b)


t = T()
# -------------- 修改属性 --------------
t.b = 10
t.test03(20)

# -------------- 修改方法01 --------------
t.test01 = lambda: print('change test01')
t.test01()


# or
def test01():
    print('change test01')


t.test01 = test01
t.test01()


# or
@torch.no_grad()
def test01(self):
    print('change test01')


t.test01 = test01
t.test01(t)

# -------------- 修改方法02 --------------
t.test02 = lambda a: print('change test02', a)
t.test02(10)

# -------------- 修改方法03 --------------
t.test03 = lambda self, a: print('change test03', a, self.b)
t.test03(t, 20)
