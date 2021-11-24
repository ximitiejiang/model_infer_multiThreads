using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenCvSharp;
using System.Runtime.InteropServices;   // Marshal
using System.IO;        // Directory


namespace mycstest_framework
{
    class AlgoTest
    {
        // 尝试直接调用model_infer.dll中的多线程api
        // 模型初始化
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static IntPtr ModelObjInit(string model_type, string model_dir, int gpu_id, bool use_trt);
        // 模型推理
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjPredict_Seg(IntPtr segObj, IntPtr imageData, int width, int height, int channels, [In, Out] IntPtr resultMap);
        // 模型资源回收
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjDestruct(IntPtr segObj);

        // 单模型多线程调用(一般为单显卡)
        public static void TestSingleModelMultiThreads()
        {
            string[] model_type_maps = { "det", "seg", "clas" }; // 支持的模型类型集合
            string model_dir = "D:\\suliang\\My_Lib11\\models\\2";
            int gpu_id = 0;
            bool use_trt = false;
            string imgdir = "D:\\suliang\\My_Dataset\\stain1k_test";
            string[] imgfiles = Directory.GetFiles(imgdir, "*.bmp");
            byte[] paddlex_model_type = new byte[10];
            // 模型
            IntPtr model = ModelObjInit(model_type_maps[1], model_dir, gpu_id, use_trt);
            int idx = 0;

            // 图像尺寸
            Mat _src = Cv2.ImRead(imgfiles[idx], ImreadModes.Grayscale); // 每个线程负责一张图
            int _h = _src.Rows;
            int _w = _src.Cols;
            IntPtr result_map1 = Marshal.AllocHGlobal(_w * _h);
            IntPtr result_map2 = Marshal.AllocHGlobal(_w * _h);
            while (idx < (imgfiles.Length / 2 * 2))
            {
                // 获取数据
                Mat src1 = Cv2.ImRead(imgfiles[idx], ImreadModes.Grayscale); // 每个线程负责一张图
                Mat src2 = Cv2.ImRead(imgfiles[idx + 1], ImreadModes.Grayscale); // 每个线程负责一张图
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
                    Console.WriteLine($"pred image id={idx + 1}");
                    ModelObjPredict_Seg(model, src2.Data, w, h, c, result_map2);
                });
                // 等待两个线程结束
                task1.Wait();
                task2.Wait();
                // 结果处理
                List<int> imgSize = new List<int>();
                imgSize.Add(w); imgSize.Add(h);
                Mat dst1 = new Mat(imgSize, MatType.CV_8UC1, result_map1);
                Mat dst2 = new Mat(imgSize, MatType.CV_8UC1, result_map2);
                Cv2.ImWrite("D:\\result_map1.bmp", dst1);
                Cv2.ImWrite("D:\\result_map2.bmp", dst2);
                // 每次两张图
                idx += 2;
                Console.WriteLine("finish 2 image pred.");
            }
            ModelObjDestruct(model);
            Marshal.FreeHGlobal(result_map1);
            Marshal.FreeHGlobal(result_map2);
        }

        // 多模型多线程(可以单显卡或者多显卡)
        public static void TestMultiModelMultiThreads()
        {
            string[] model_type_maps = { "det", "seg", "clas" }; // 支持的模型类型集合
            string model_dir = "D:\\suliang\\My_Lib11\\models\\2";
            int gpu_id = 0;
            bool use_trt = false;
            string imgdir = "D:\\suliang\\My_Dataset\\stain1k_test";
            string[] imgfiles = Directory.GetFiles(imgdir, "*.bmp");
            byte[] paddlex_model_type = new byte[10];
            // 模型
            IntPtr model1 = ModelObjInit(model_type_maps[1], model_dir, gpu_id, use_trt);
            IntPtr model2 = ModelObjInit(model_type_maps[1], model_dir, gpu_id, use_trt);
            int idx = 0;

            // 图像尺寸
            Mat _src = Cv2.ImRead(imgfiles[idx], ImreadModes.Grayscale);
            int _h = _src.Rows;
            int _w = _src.Cols;
            IntPtr result_map1 = Marshal.AllocHGlobal(_w * _h);
            IntPtr result_map2 = Marshal.AllocHGlobal(_w * _h);
            IntPtr result_map3 = Marshal.AllocHGlobal(_w * _h);
            IntPtr result_map4 = Marshal.AllocHGlobal(_w * _h);
            while (idx < (imgfiles.Length / 4 * 4))
            {
                // 获取数据
                Mat src1 = Cv2.ImRead(imgfiles[idx], ImreadModes.Grayscale); // 每个线程负责一张图
                Mat src2 = Cv2.ImRead(imgfiles[idx + 1], ImreadModes.Grayscale); // 每个线程负责一张图
                Mat src3 = Cv2.ImRead(imgfiles[idx + 2], ImreadModes.Grayscale); // 每个线程负责一张图
                Mat src4 = Cv2.ImRead(imgfiles[idx + 3], ImreadModes.Grayscale); // 每个线程负责一张图
                int h = src1.Rows;
                int w = src1.Cols;
                int c = src1.Channels();

                // 模型1的多线程访问
                Task task1 = Task.Run(() =>
                {
                    Console.WriteLine($"pred image id={idx}");
                    ModelObjPredict_Seg(model1, src1.Data, w, h, c, result_map1);
                });
                Task task2 = Task.Run(() =>
                {
                    Console.WriteLine($"pred image id={idx + 1}");
                    ModelObjPredict_Seg(model1, src2.Data, w, h, c, result_map2);
                });
                // 模型2的多线程访问
                Task task3 = Task.Run(() =>
                {
                    Console.WriteLine($"pred image id={idx + 2}");
                    ModelObjPredict_Seg(model2, src3.Data, w, h, c, result_map3);
                });
                Task task4 = Task.Run(() =>
                {
                    Console.WriteLine($"pred image id={idx + 3}");
                    ModelObjPredict_Seg(model2, src4.Data, w, h, c, result_map4);
                });
                // 等待两个线程结束
                task1.Wait();
                task2.Wait();
                task3.Wait();
                task4.Wait();
                // 结果处理
                List<int> imgSize = new List<int>();
                imgSize.Add(w); imgSize.Add(h);
                Mat dst1 = new Mat(imgSize, MatType.CV_8UC1, result_map1);
                Mat dst2 = new Mat(imgSize, MatType.CV_8UC1, result_map2);
                Mat dst3 = new Mat(imgSize, MatType.CV_8UC1, result_map3);
                Mat dst4 = new Mat(imgSize, MatType.CV_8UC1, result_map4);
                Cv2.ImWrite("D:\\result_map1.bmp", dst1);
                Cv2.ImWrite("D:\\result_map2.bmp", dst2);
                Cv2.ImWrite("D:\\result_map3.bmp", dst3);
                Cv2.ImWrite("D:\\result_map4.bmp", dst4);
                // 每次两张图
                idx += 4;
                Console.WriteLine("finish 4 image pred.");
            }
            ModelObjDestruct(model1);
            ModelObjDestruct(model2);
            Marshal.FreeHGlobal(result_map1);
            Marshal.FreeHGlobal(result_map2);
            Marshal.FreeHGlobal(result_map3);
            Marshal.FreeHGlobal(result_map4);
        }

    }
    class Program
    {
        static void Main(string[] args)
        {
            //AlgoTest.TestSingleModelMultiThreads();

            AlgoTest.TestMultiModelMultiThreads();

            Console.WriteLine("finished pred");
            Console.ReadLine();
        }
    }
}