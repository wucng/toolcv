#ifndef	_MY_OPENCV_H
#define _MY_OPENCV_H 1

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "myhelp.h"

using namespace std;
using namespace cv;

/**copy*/
int imgCopy(string &imgPath)
{
    Mat src = imread(imgPath,cv::IMREAD_COLOR);
    if (src.empty())// !src.data
    {
        cout<<"fail"<<endl;
        return -1;
    }
    int h,w,c;
    h = src.rows;
    w = src.cols;
    c = src.channels();

    // 操作数据指针实现复制功能 
    // Mat dst;
    // Mat dst = Mat::zeros(h,w,CV_8UC(c));
    Mat dst = Mat::zeros(Size(w,h),CV_8UC(c));
    // Mat dst = Mat::zeros(src.size(),src.type());
    auto ptr_src = src.data; // src首地址指针
    auto ptr_dst = dst.data;
    for (int i=0;i<h;++i)
    {
        for (int j=0;j<w;++j)
        {
            ptr_dst[i*w*c+j*c+0]=ptr_src[i*w*c+j*c+0];
            ptr_dst[i*w*c+j*c+1]=ptr_src[i*w*c+j*c+1];
            ptr_dst[i*w*c+j*c+2]=ptr_src[i*w*c+j*c+2];
        }
    }

    // 保存
    cv::imwrite("copy.jpg",dst);
    return 0;
}

void imgCopy(cv::Mat &src,cv::Mat &dst)
{
    int h,w,c;
    h = src.rows;
    w = src.cols;
    c = src.channels();

    uchar *ptr_src = src.data; // src首地址指针
    uchar *ptr_dst = dst.data;

    for (int i=0;i<h;++i) // h x w x c 格式存储
    {
        for (int j=0;j<w;++j)
        {
            ptr_dst[i*w*c+j*c+0]=ptr_src[i*w*c+j*c+0];
            ptr_dst[i*w*c+j*c+1]=ptr_src[i*w*c+j*c+1];
            ptr_dst[i*w*c+j*c+2]=ptr_src[i*w*c+j*c+2];
        }
    }
}


void loadData(std::string imgPath,float **hostDataBuffer)
{
        float mean[]{0.485,0.456,0.406};
        float std[]{0.229,0.224,0.225};
        //read img
        cv::Mat img = cv::imread(imgPath,cv::IMREAD_COLOR);
        // resize
        cv::Mat res_img;
        cv::resize(img,res_img,cv::Size(256,256),0,0,cv::INTER_LINEAR);
        // center crop
        cv::Mat crop_img(res_img,cv::Rect(16,16,224,224));
        // BGR->RGB
        cv::Mat rgb_img;
        cv::cvtColor(crop_img,rgb_img,cv::COLOR_BGR2RGB);
       
        int h,w,c;
        h = rgb_img.rows;
        w = rgb_img.cols;
        c = rgb_img.channels();

        auto ptr_src = rgb_img.data;

         // normalnize and h,w,c-->c,h,w
        int ih{0},iw{0},ic{0};
        for (int i=0;i<c*h*w;++i)
        {
                ic = i/(h*w);
                ih = i%(h*w)/w;
                iw = i%(h*w)%w;
                (*hostDataBuffer)[i] = (ptr_src[ih*(w*c)+iw*c+ic]/255.0f-mean[ic])/std[ic];
        }

}

void loadData(std::string imgPath,float **hostDataBuffer, int batchIdx)
{
        float mean[]{0.485,0.456,0.406};
        float std[]{0.229,0.224,0.225};
        //read img
        cv::Mat img = cv::imread(imgPath,cv::IMREAD_COLOR);
        // resize
        cv::Mat res_img;
        cv::resize(img,res_img,cv::Size(256,256),0,0,cv::INTER_LINEAR);
        // center crop
        cv::Mat crop_img(res_img,cv::Rect(16,16,224,224));
        // BGR->RGB
        cv::Mat rgb_img;
        cv::cvtColor(crop_img,rgb_img,cv::COLOR_BGR2RGB);
       
        int h,w,c;
        h = rgb_img.rows;
        w = rgb_img.cols;
        c = rgb_img.channels();

        auto ptr_src = rgb_img.data;

         // normalnize and h,w,c-->c,h,w
        int ih{0},iw{0},ic{0};
        for (int i=0;i<c*h*w;++i)
        {
                ic = i/(h*w);
                ih = i%(h*w)/w;
                iw = i%(h*w)%w;
                (*hostDataBuffer)[batchIdx*(c*h*w)+i] = (ptr_src[ih*(w*c)+iw*c+ic]/255.0f-mean[ic])/std[ic];
        }

}

void loadData(std::string imgPath,float *hostDataBuffer, int batchIdx)
{
        float mean[]{0.485,0.456,0.406};
        float std[]{0.229,0.224,0.225};
        //read img
        cv::Mat img = cv::imread(imgPath,cv::IMREAD_COLOR);
        // resize
        cv::Mat res_img;
        cv::resize(img,res_img,cv::Size(256,256),0,0,cv::INTER_LINEAR);
        // center crop
        cv::Mat crop_img(res_img,cv::Rect(16,16,224,224));
        // BGR->RGB
        cv::Mat rgb_img;
        cv::cvtColor(crop_img,rgb_img,cv::COLOR_BGR2RGB);
       
        int h,w,c;
        h = rgb_img.rows;
        w = rgb_img.cols;
        c = rgb_img.channels();

        auto ptr_src = rgb_img.data;

         // normalnize and h,w,c-->c,h,w
        int ih{0},iw{0},ic{0};
        for (int i=0;i<c*h*w;++i)
        {
                ic = i/(h*w);
                ih = i%(h*w)/w;
                iw = i%(h*w)%w;
                hostDataBuffer[batchIdx*(c*h*w)+i] = (ptr_src[ih*(w*c)+iw*c+ic]/255.0f-mean[ic])/std[ic];
        }

}

void pprintResult(float **output,const int &outputSize)
{
        #if 0
        {
                for (int i = 0; i < outputSize; i++)
                {
                        std::cout<<(*output)[i]<<",";
                }
                std::cout<<std::endl;
        }
        
        #else
        {
                float val{0.0f};
                int idx{0};
                // Calculate Softmax
                float sum{0.0f};
                for (int i = 0; i < outputSize; i++)
                {
                        (*output)[i] = exp((*output)[i]);
                        sum += (*output)[i];
                }

                mycout << "Output:" << std::endl;
                for (int i = 0; i < outputSize; i++)
                {
                        (*output)[i] /= sum;
                        val = std::max(val, (*output)[i]);
                        if (val == (*output)[i])
                        {
                                idx = i;
                        }

                        // mycout << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                                //  << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
                }

                mycout<<"score: "<< std::fixed << std::setw(5) << std::setprecision(4) <<val<<" Class:"<<idx<<endl;

                mycout << std::endl;
        }
        #endif
}

void pprintResult(float **output,const int &outputSize,int &count,vector<float> &scores,vector<int> &labels)
{
        float val{0.0f};
        int idx{0};
        // Calculate Softmax
        float sum{0.0f};
        for (int j=0;j<count;++j)
        {
                val = 0.0f;
                idx = 0;
                sum = 0.0f;
                for (int i = 0; i < outputSize; i++)
                {
                        (*output)[j*outputSize+i] = exp((*output)[j*outputSize+i]);
                        sum += (*output)[j*outputSize+i];
                }

                mycout << "Output:" << std::endl;
                for (int i = 0; i < outputSize; i++)
                {
                        (*output)[j*outputSize+i] /= sum;
                        val = std::max(val, (*output)[j*outputSize+i]);
                        if (val == (*output)[j*outputSize+i])
                        {
                                idx = i;
                        }

                        // mycout << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                                //  << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
                }

                mycout<<"score: "<< std::fixed << std::setw(5) << std::setprecision(4) <<val<<" Class:"<<idx<<endl;

                // mycout << std::endl;

                scores.push_back(val);
                labels.push_back(idx);
        }
        

}


#endif /*myopencv.h*/
