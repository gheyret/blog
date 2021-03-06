Eager执行基础

[原文地址](https://tensorflow.google.cn/tutorials/eager/eager_basics)

这是使用TensorFlow的入门教程。 它将涵盖：  

1. 导入所需的包
2. 创建和使用张量
3. 使用GPU加速
4. 数据集

## Import TensorFlow ##

首先，导入tensorflow模块并启用eager execution。 eager execution为TensorFlow提供了一个更加互动的前端，我们将在稍后讨论其中的详细信息。

    import tensorflow as tf
    
    tf.enable_eager_execution()

### Tensors ###

张量是一个多维数组。 与NumPy ndarray对象类似，Tensor对象具有数据类型和形状。 此外，Tensors可以驻留在加速器（如GPU）内存中。 TensorFlow提供了丰富的操作库（tf.add，tf.matmul，tf.linalg.inv等），它们使用和生成Tensors。 这些操作自动转换原生Python类型。 例如：

    print(tf.add(1, 2))
    print(tf.add([1, 2], [3, 4]))
    print(tf.square(5))
    print(tf.reduce_sum([1, 2, 3]))
    print(tf.encode_base64("hello world"))
    
    # Operator overloading is also supported
    print(tf.square(2) + tf.square(3))

*tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor([4 6], shape=(2,), dtype=int32)
tf.Tensor(25, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(b'aGVsbG8gd29ybGQ', shape=(), dtype=string)
tf.Tensor(13, shape=(), dtype=int32)*

每个张量都有一个形状和数据类型：

    x = tf.matmul([[1]], [[2, 3]])
    print(x.shape)
    print(x.dtype)

*(1, 2)
<dtype: 'int32'>*

NumPy阵列和TensorFlow张量之间最明显的区别是：

1. 张量可以由加速器内存（如GPU，TPU）支持。
2. 张量是不可改变的。

### NumPy兼容性 ###

TensorFlow张量和NumPy nararrays之间的转换非常简单，如：

TensorFlow操作自动将NumPy ndarrays转换为Tensors。
NumPy操作自动将Tensors转换为NumPy ndarrays。
通过在它们上调用.numpy（）方法，可以将张量显式转换为NumPy ndarrays。 这些转换通常很便宜，因为如果可能，数组和Tensor共享底层内存表示。 但是，共享底层表示并不总是可行的，因为Tensor可能托管在GPU内存中，而NumPy阵列总是由主机内存支持，因此转换将涉及从GPU到主机内存的复制。

    import numpy as np
    
    ndarray = np.ones([3, 3])
    
    print("TensorFlow operations convert numpy arrays to Tensors automatically")
    tensor = tf.multiply(ndarray, 42)
    print(tensor)
    
    print("And NumPy operations convert Tensors to numpy arrays automatically")
    print(np.add(tensor, 1))
    
    print("The .numpy() method explicitly converts a Tensor to a numpy array")
    print(tensor.numpy())

*TensorFlow operations convert numpy arrays to Tensors automatically
tf.Tensor(
[[42. 42. 42.]
 [42. 42. 42.]
 [42. 42. 42.]], shape=(3, 3), dtype=float64)
And NumPy operations convert Tensors to numpy arrays automatically
[[43. 43. 43.]
 [43. 43. 43.]
 [43. 43. 43.]]
The .numpy() method explicitly converts a Tensor to a numpy array
[[42. 42. 42.]
 [42. 42. 42.]
 [42. 42. 42.]]*

### GPU 加速 ###

通过使用GPU进行计算，可以加速许多TensorFlow操作。 在没有任何注释的情况下，TensorFlow会自动决定是使用GPU还是CPU进行操作（如有必要，还可以复制CPU和GPU内存之间的张量）。 由操作产生的张量通常由执行操作的设备的存储器支持。 例如：

    x = tf.random_uniform([3, 3])
    
    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    
    print("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'))

*Is there a GPU available: 
False
Is the Tensor on GPU #0:  
False*

### 设备名称 ###

Tensor.device属性提供托管张量内容的设备的完全限定字符串名称。 此名称编码许多详细信息，例如正在执行此程序的主机的网络地址的标识符以及该主机中的设备。 这是分布式执行TensorFlow程序所必需的。 如果张量位于主机上的第N个GPU上，则字符串以GPU结尾：<N>。

### 显式设备放置 ###

TensorFlow中的术语“placement”指的是如何为执行设备分配（放置）各个操作。 如上所述，当没有提供明确的指导时，TensorFlow会自动决定执行操作的设备，并在需要时将Tensors复制到该设备。 但是，可以使用tf.device上下文管理器将TensorFlow操作显式放置在特定设备上。 例如：

    def time_matmul(x):
      %timeit tf.matmul(x, x)
    
    # Force execution on CPU
    print("On CPU:")
    with tf.device("CPU:0"):
      x = tf.random_uniform([1000, 1000])
      assert x.device.endswith("CPU:0")
      time_matmul(x)
    
    # Force execution on GPU #0 if available
    if tf.test.is_gpu_available():
      with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

*On CPU:
7.01 ms ± 362 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)*

## Datasets ##

本节演示如何使用tf.data.Dataset API构建管道以将数据提供给模型。 它涵盖：

1. 创建数据集。
2. 在启用了急切执行的情况下对数据集进行迭代。

我们建议使用数据集API从简单，可重复使用的部分构建高性能，复杂的输入管道，这些部分将为模型的培训或评估循环提供支持。

如果您熟悉TensorFlow图，则在启用eager执行时，构建数据集对象的API保持完全相同，但迭代数据集元素的过程稍微简单一些。 您可以对tf.data.Dataset对象使用Python迭代，而不需要显式创建tf.data.Iterator对象。 因此，在启用eager执行时，TensorFlow指南中对迭代器的讨论无关紧要。

### Create a source Dataset ###

使用其中一个工厂函数（如Dataset.from_tensors，Dataset.from_tensor_slices）或使用从TextLineDataset或TFRecordDataset等文件读取的对象创建源数据集。 有关更多信息，请参阅TensorFlow指南。

    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    
    # Create a CSV file
    import tempfile
    _, filename = tempfile.mkstemp()
    
    with open(filename, 'w') as f:
      f.write("""Line 1
    Line 2
    Line 3
      """)
    
    ds_file = tf.data.TextLineDataset(filename)

### 应用转换 ###

使用map，batch，shuffle等转换函数将转换应用于数据集的记录。 有关详细信息，请参阅tf.data.Dataset的API文档。

    ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
    ds_file = ds_file.batch(2)

### 迭代 ###

启用eager执行时，Dataset对象支持迭代。 如果您熟悉TensorFlow图中数据集的使用，请注意不需要调用Dataset.make_one_shot_iterator（）或get_next（）调用。

    print('Elements of ds_tensors:')
    for x in ds_tensors:
      print(x)
    
    print('\nElements in ds_file:')
    for x in ds_file:
      print(x)

*Elements of ds_tensors:
tf.Tensor([4 1], shape=(2,), dtype=int32)
tf.Tensor([16 25], shape=(2,), dtype=int32)
tf.Tensor([36  9], shape=(2,), dtype=int32)

Elements in ds_file:
tf.Tensor([b'Line 1' b'Line 2'], shape=(2,), dtype=string)
tf.Tensor([b'Line 3' b'  '], shape=(2,), dtype=string)*