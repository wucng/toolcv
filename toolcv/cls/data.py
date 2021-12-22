import os
import gzip
import io
import time
import urllib3
import requests
import numpy as np

def download_file(url,name,save_path='./data'):
    if not os.path.exists(save_path):os.makedirs(save_path)
    save_path = os.path.join(save_path,name)
    if os.path.exists(save_path):
        print("%s exists"%save_path)
        return
    # http = requests.get(os.path.join(url,name))
    # with open(save_path, 'wb') as f:
    #     f.write(http.content)

    http = urllib3.PoolManager()
    response = http.request('GET', os.path.join(url, name))
    with open(save_path, 'wb') as f:
        f.write(response.data)

    print("----download successful----")

def load_data(read_file_name,offset=0):
    with gzip.open(read_file_name, 'rb') as input_file:
        # with io.TextIOWrapper(input_file, encoding='utf-8') as dec:
        #     data = dec.read()
        #     print(data)
        # np.fromfile()
        data = np.frombuffer(input_file.read(),dtype='uint8')
        return data[offset:]

def load_mnist(save_path = './data',mode="mnist"):
    if mode == "mnist":
        url = "http://yann.lecun.com/exdb/mnist/"
    if mode == "fashion-mnist":
        url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    train_images = "train-images-idx3-ubyte.gz"
    train_labels = "train-labels-idx1-ubyte.gz"
    test_images = "t10k-images-idx3-ubyte.gz"
    test_labels = "t10k-labels-idx1-ubyte.gz"

    download_file(url, test_images, save_path)
    download_file(url, test_labels, save_path)
    download_file(url, train_images, save_path)
    download_file(url, train_labels, save_path)

    train_labels = load_data(os.path.join(save_path, train_labels), 8)
    train_images = load_data(os.path.join(save_path, train_images), 16)
    train_images = train_images.reshape(-1, 28, 28) # 0~255

    test_labels = load_data(os.path.join(save_path, test_labels), 8)
    test_images = load_data(os.path.join(save_path, test_images), 16)
    test_images = test_images.reshape(-1, 28, 28) # 0~255

    return train_images,train_labels,test_images,test_labels

if __name__ == "__main__":
    train_images,train_labels,test_images,test_labels = load_mnist(mode='fashion-mnist')
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)