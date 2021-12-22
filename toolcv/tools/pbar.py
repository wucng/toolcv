import time
import sys
from tqdm import tqdm

def progressbar(desc:str,i, total_len,len_pbar=20):
    s = "[" + "#" * int(i * len_pbar / total_len) + " " * (len_pbar - int(i * len_pbar / total_len)) + "]"
    s = "\r%s %s:%.2f%%"%(desc,s,1.0*i / total_len*100)
    # print(s,end="",flush=True,file=open("123.txt",'w'))
    print(s,end="",flush=True)
    # sys.stdout.write(s)
    # return s

class Progressbar:
    def __init__(self,pbar,total_len=None):
        self.pbar = iter(pbar)
        self.idx = 0
        self.total_len = len(pbar) if total_len is None else total_len

    def __iter__(self):
        while True:
            yield next(self.pbar)
            self.idx += 1
            if self.idx >= self.total_len:break

    def set_description(self,desc):
        progressbar(desc,self.idx+1,self.total_len)

if __name__ == "__main__":
    # fp = open("123.txt", 'w')
    # for i in range(300):
    #     time.sleep(0.01)
    #     progressbar("",i+1,300)
        # print(i, flush=True, file=fp)
    # fp.close()
    pbar = Progressbar(enumerate(range(300)),300)
    for i,j in pbar:
        time.sleep(0.01)
        pbar.set_description(str(i)+"进度条")