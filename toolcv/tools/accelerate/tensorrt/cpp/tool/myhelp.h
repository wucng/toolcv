#ifndef	_MY_HELP_H
#define _MY_HELP_H 1

/**
 * 封装一些常用函数，方便后续使用
*/

// #include "NvInfer.h"
// #include "cuda_runtime_api.h"
// #include "common.h"
#include <memory>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <string>
#include <cstring>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

// using namespace std;

std::string getCurrentTime();

#define mycout std::cout<<"["<<__FILE__<<":"<<__LINE__<< ":" << getCurrentTime() <<"] "

/**获取当前系统时间 如：2020-07-31 09:50:53 */
std::string getCurrentTime()
{
        /**获取当前系统时间 如：2020-07-31 09:50:53 */
        struct tm t;   //tm结构指针
        time_t now;  //声明time_t类型变量
        time(&now);      //获取系统日期和时间
        t = *localtime(&now);   //获取当地日期和时间

        char szResult[20] = "\0";
        sprintf(szResult,"%.4d-%.2d-%.2d %.2d:%.2d:%.2d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min,t.tm_sec);
        // std::cout << szResult << std::endl;
        return std::string(szResult);
}

/**定义常用的初始化参数*/
class Params
{
    public:
        std::string imgPath ;  // 图片路径
        // 查找是否已经存在trt文件
        std::string engineFile; //= "./serialize_engine_output.trt";
        std::string wtsFile; //!< Filename of wts file of a network

        // stuff we know about the network and the input/output blobs
        int INPUT_H = 224;
        int INPUT_W = 224;
        int OUTPUT_SIZE = 1000;
        int BATCH_SIZE = 32;

        bool int8{false};                  //!< Allow runnning the network in Int8 mode.
        bool fp16{false};                  //!< Allow running the network in FP16 mode.

        char* INPUT_BLOB_NAME = "input";
        char* OUTPUT_BLOB_NAME = "output";

    public:
        Params(std::string imgPath,std::string wtsFile,std::string engineFile,std::string mode);
};

/**构造函数定义*/
inline Params::Params(std::string imgPath,std::string wtsFile,std::string engineFile,std::string mode):
imgPath(imgPath),wtsFile(wtsFile),engineFile(engineFile)
{
    if(mode=="fp16")
            fp16=true;
    else if (mode=="int8")
            int8=true;
}


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

/**智能指针，会指定销毁指针 释放内存*/
template<typename T>
std::shared_ptr<T> smart_ptr(T *pointer)
{
    return std::shared_ptr<T>(pointer,InferDeleter());// pointer实现了destroy()方法
}

template<typename T>
std::unique_ptr<T> unique_smart_ptr(T *pointer)
{
    return std::unique_ptr<T>(pointer);
}

template <typename T> // 智能指针封装一层
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;// unique_ptr 智能指针（可以自动清除创建的对象）
// 相当于 下面的重命名，但是 typedef 不能使用模板，必须指定具体类型
// template <typename T> 
// typedef std::unique_ptr<T, InferDeleter> SampleUniquePtr; 


/**获取指定格式的目录文件*/
void  glob_format(std::vector<std::string>& paths, char* dir,std::vector<const char *>suffixNames ,int depth=1,std::string all_dir="") //=".jpg"
{
    DIR *p_dir = nullptr;
    struct dirent *p_entry=nullptr;
    struct stat statbuf;

    if ((p_dir=opendir(dir))==NULL)
    {
        printf("can't open %s.\n",dir);
        return;
    }

    std::string tmp;

    if (all_dir=="")
        all_dir=std::string(dir);
        // strcpy(all_dir,dir);

    // if(all_dir[strlen(all_dir)-1]!='/')
        //    strcat(all_dir, "/"); // strcpy
    if(all_dir[all_dir.length()-1]!='/')
        all_dir.push_back('/');


    chdir(dir);

    while (NULL != (p_entry=readdir(p_dir)))// 获取下一级目录信息
    {
        
        lstat(p_entry->d_name,&statbuf);// 获取下一级成员属性

        if(S_IFDIR & statbuf.st_mode){ // 判断下一级成员是否是目录  
            if (strcmp(".",p_entry->d_name)==0 || strcmp("..",p_entry->d_name)==0) // 对应系统文件
                continue;
             
            //  printf("%*s%s/\n", depth, "", p_entry->d_name);
             tmp = all_dir+std::string(p_entry->d_name);
            glob_format(paths,p_entry->d_name, suffixNames,depth+4,tmp); // 扫描下一级目录的内容
            tmp ="";
        }
        else 
        {
            // printf("%*s%s\n", depth, "", p_entry->d_name); // 输出属性不是目录的成员
            // sprintf(tmp,"%s%s",(const char *)all_dir,(const char *)p_entry->d_name);
            // strcat(tmp,(const char *)all_dir);
            // strcat(tmp,p_entry->d_name);
            for (auto suffixName:suffixNames)
            {
                    if (strstr(p_entry->d_name,suffixName)) // 如果是.jpg后缀的
                    {
                        tmp = all_dir+std::string(p_entry->d_name);

                        paths.push_back(tmp);
                        tmp ="";

                        break;
                    }
            }
            
             
        }
    }
    
    chdir(".."); // 回到上级目录  
    closedir(p_dir);
}


#endif /* myhelp.h.  */
