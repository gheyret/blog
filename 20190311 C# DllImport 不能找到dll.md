# C#  DllImport 相对路径无法找到dll #

如题，近期在开发过程中，需要调用C++的库，一般来说，使用下面的方法即可正常调用：

    [DllImport("hci_sys.dll", CallingConvention = CallingConvention.StdCall)]
    public static extern int hci_init(string initConfig);

这个只要的需要的库文件放到根目录就可以了，但如果啥文件都往根目录放，那就显得比较杂乱无章，而且不好维护，所以通常会建立不同的目录放在不同的文件，这样程序目录结构就比较清晰。所以我就调整了目录结构，然后重新调用C++的方法如下：

        [DllImport(@"\AsrSdk\Jths\hci_sys.dll", CallingConvention = CallingConvention.StdCall)]
        public static extern int hci_init(string initConfig);

把库文件都放到了根目录的相对路径下面，按理说应该也可以正常运行的，BUT，事情总不会按设定好的剧情发展，因为调用的C++ sdk本身包含多个C++的dll，我调用的这个接口是成功了，但是其内部又用到了它自己的dll，导致找不到它自己的dll，造成调用失败。

查阅了很多资料，因为C++的代码中，Windows目录和Windows\System32目录以及环境变量设定的目录都是搜索路径之一，所以最后的解决办法就是动态设置当前进程的环境变量，把sdk所在的目录添加到环境变量中，这其内部就可以找到自己的dll了：

        /// <summary>
        /// 添加环境变量
        /// </summary>
        /// <param name="paths">路径列表</param>
        internal static void AddEnvironmentPaths(IEnumerable<string> paths)
        {          
            var path = new[] { Environment.GetEnvironmentVariable("PATH") ?? string.Empty };
            string newPath = string.Join(Path.PathSeparator.ToString(), path.Concat(paths));
            Environment.SetEnvironmentVariable("PATH", newPath);   // 这种方式只会修改当前进程的环境变量
        }














