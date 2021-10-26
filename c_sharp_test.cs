using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AlgCommon;
using Models;
using OpenCvSharp;
using System.Runtime.InteropServices;   // Marshal
using System.IO;        // Directory
using System.Drawing;   // 需要手动添加引用System.Drawing
using PixelFormat = System.Drawing.Imaging.PixelFormat;
using System.Threading;

namespace mycstest_framework
{
    class AlgoTest
    {
        // 模型初始化
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static IntPtr ModelObjInit(string model_type, string model_dir, int gpu_id, bool use_trt);
        // 模型推理
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjPredict_Seg(IntPtr segObj, IntPtr imageData, int width, int height, int channels, [In, Out] IntPtr resultMap);
        // 模型资源回收
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjDestruct(IntPtr segObj);
        public static void TestMultithread()
        {
            string[] model_type_maps = { "det", "seg", "clas" }; // 支持的模型类型集合
            string model_dir = "D:\\ClassificationPlatform\\WafetDetect.Algorithm\\lib\\models\\1";
            int gpu_id = 0;
            bool use_trt = false;
            string imgdir = "D:\\My_Dataset\\test10_1024_bmpc1";
            string[] imgfiles = Directory.GetFiles(imgdir, "*.bmp");
            byte[] paddlex_model_type = new byte[10];
            // 模型
            IntPtr model = ModelObjInit(model_type_maps[1], model_dir, gpu_id, use_trt);
            int idx = 0;

            // 图像尺寸
            Mat _src = Cv2.ImRead(imgfiles[idx], ImreadModes.Grayscale); 
            int _h = _src.Rows;
            int _w = _src.Cols;
            IntPtr result_map1 = Marshal.AllocHGlobal(_w * _h);
            IntPtr result_map2 = Marshal.AllocHGlobal(_w * _h);
            while (idx < (imgfiles.Length/2*2))
            {
                // 获取数据
                Mat src1 = Cv2.ImRead(imgfiles[idx], ImreadModes.Grayscale); // 每个线程负责一张图
                Mat src2 = Cv2.ImRead(imgfiles[idx+1], ImreadModes.Grayscale); // 每个线程负责一张图
                int h = src1.Rows;
                int w = src1.Cols;
                int c = src1.Channels();

                // 启动2个线程同时访问模型也不会冲突
                Task task1 = Task.Run(() =>
                {
                    Console.WriteLine($"pred image id={idx}");
                    ModelObjPredict_Seg(model, src1.Data, w, h, c, result_map1);
                });
                Task task2 = Task.Run(() =>
                {
                    Console.WriteLine($"pred image id={idx+1}");
                    ModelObjPredict_Seg(model, src2.Data, w, h, c, result_map2);
                });
                // 等待两个线程结束
                task1.Wait();
                task2.Wait();
                // 结果处理
                List<int> imgSize = new List<int>();
                imgSize.Add(w);imgSize.Add(h);
                Mat dst1 = new Mat(imgSize, MatType.CV_8UC1, result_map1);
                Mat dst2 = new Mat(imgSize, MatType.CV_8UC1, result_map2);
                Cv2.ImWrite("D:\\result_map1.bmp", dst1);
                Cv2.ImWrite("D:\\result_map2.bmp", dst2);
                // 每次两张图
                idx +=2;
                Console.WriteLine("finish 2 image pred.");
            }
            ModelObjDestruct(model);
            Marshal.FreeHGlobal(result_map1);
            Marshal.FreeHGlobal(result_map2);
        }

    }
    class Program
    {
        static void Main(string[] args)
        {
            AlgoTest.TestMultithread();
            Console.WriteLine("finished pred");
            Console.ReadLine();
        }
    }
}
