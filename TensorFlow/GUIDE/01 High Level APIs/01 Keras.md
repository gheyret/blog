
[原文地址](https://tensorflow.google.cn/guide/keras)

Keras是一个用于构建和训练深度学习模型的高级API。 它用于快速原型设计，高级研究和生产，具有三个主要优势：

1. 方便使用的  
Keras具有针对常见用例优化的简单，一致的界面。 它为用户错误提供清晰且可操作的反馈。
2. 模块化和可组合  
Keras模型是通过将可配置的构建块连接在一起而制定的，几乎没有限制。
3. 易于扩展  
编写自定义构建块以表达研究的新想法。 创建新图层，损失函数并开发最先进的模型。

## Import tf.keras ##

tf.keras是TensorFlow实现的Keras API规范。 这是一个用于构建和训练模型的高级API，其中包括对TensorFlow特定功能的一流支持，例如eager execution，tf.data管道和Estimators。 tf.keras使TensorFlow更易于使用，而不会牺牲灵活性和性能。

要开始，请导入tf.keras作为TensorFlow程序设置的一部分：

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

*1.11.0  
2.1.6-tf*

tf.keras可以运行任何与Keras兼容的代码，但请记住：

最新TensorFlow版本中的tf.keras版本可能与PyPI的最新keras版本不同。 检查tf.keras.version。
保存模型的权重时，tf.keras默认为检查点格式。 通过save_format ='h5'使用HDF5。

## Buile a simple model ##

### Sequential model ###

在Keras中，您可以组装图层来构建模型。 模型（通常）是图层图。 最常见的模型类型是一堆层：tf.keras.Sequential模型。

构建一个简单的，完全连接的网络（即多层感知器）：

    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(10, activation='softmax'))

### Configure the layers ###

有许多tf.keras.layers可用于一些常见的构造函数参数：

1. activation：设置图层的激活功能。 此参数由内置函数的名称或可调用对象指定。 默认情况下，不应用任何激活。
2. kernel_initializer和bias_initializer：创建层权重（内核和偏差）的初始化方案。 此参数是名称或可调用对象。 这默认为“Glorot uniform”初始化程序。
3. kernel_regularizer和bias_regularizer：应用层权重（内核和偏差）的正则化方案，例如L1或L2正则化。 默认情况下，不应用正则化。 
 
下面使用构造函数参数实例化tf.keras.layers.Dense图层：

    # Create a sigmoid layer:
    layers.Dense(64, activation='sigmoid')
    # Or:
    layers.Dense(64, activation=tf.sigmoid)
    
    # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
    layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
    
    # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
    layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
    
    # A linear layer with a kernel initialized to a random orthogonal matrix:
    layers.Dense(64, kernel_initializer='orthogonal')
    
    # A linear layer with a bias vector initialized to 2.0s:
    layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

## Train and evaluate ##

### Set up training ###

构建模型后，通过调用compile方法配置其学习过程：

    model = tf.keras.Sequential([
    # Adds a densely-connected layer with 64 units to the model:
    layers.Dense(64, activation='relu'),
    # Add another:
    layers.Dense(64, activation='relu'),
    # Add a softmax layer with 10 output units:
    layers.Dense(10, activation='softmax')])
    
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])

tf.keras.Model.compile有三个重要参数：

1. optimizer：此对象指定训练过程。 从tf.train模块传递优化器实例，例如tf.train.AdamOptimizer，tf.train.RMSPropOptimizer或tf.train.GradientDescentOptimizer。
2. loss：在优化期间最小化的函数。 常见的选择包括均方误差（mse），categorical_crossentropy和binary_crossentropy。 损失函数由名称或通过从tf.keras.losses模块传递可调用对象来指定。
3. metrics：用于监控训练。 这些是字符串名称或来自tf.keras.metrics的可调用模块。

以下显示了配置训练模型的几个示例：

    # Configure a model for mean-squared error regression.
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
      loss='mse',   # mean squared error
      metrics=['mae'])  # mean absolute error
    
    # Configure a model for categorical classification.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
      loss=tf.keras.losses.categorical_crossentropy,
      metrics=[tf.keras.metrics.categorical_accuracy])

### Input NumPy data ###

对于小型数据集，请使用内存中的NumPy阵列来训练和评估模型。 使用拟合方法将模型“拟合”到训练数据：

    import numpy as np
    
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    
    model.fit(data, labels, epochs=10, batch_size=32)

*Epoch 1/10
1000/1000 [==============================] - 0s 253us/step - loss: 11.5766 - categorical_accuracy: 0.1110
Epoch 2/10
1000/1000 [==============================] - 0s 64us/step - loss: 11.5205 - categorical_accuracy: 0.1070
Epoch 3/10
1000/1000 [==============================] - 0s 70us/step - loss: 11.5146 - categorical_accuracy: 0.1100
Epoch 4/10
1000/1000 [==============================] - 0s 69us/step - loss: 11.5070 - categorical_accuracy: 0.0940
Epoch 5/10
1000/1000 [==============================] - 0s 71us/step - loss: 11.5020 - categorical_accuracy: 0.1150
Epoch 6/10
1000/1000 [==============================] - 0s 72us/step - loss: 11.5019 - categorical_accuracy: 0.1350
Epoch 7/10
1000/1000 [==============================] - 0s 72us/step - loss: 11.5012 - categorical_accuracy: 0.0970*

tf.keras.Model.fit有三个重要参数：

1. epochs：训练为周期性的。 一个周期是对整个输入数据的一次迭代（这是以较小的批次完成的）。
2. batch_size：当传递NumPy数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。 此整数指定每个批次的大小。 请注意，如果样本总数不能被批量大小整除，则最后一批可能会更小。
3. validation_data：在对模型进行原型设计时，您希望轻松监控其在某些验证数据上的性能。 传递这个参数 - 输入和标签的元组 - 允许模型在每个周期的末尾以传递数据的推理模式显示损失和度量。

这是使用validation_data的示例：

    import numpy as np
    
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    
    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))
    
    model.fit(data, labels, epochs=10, batch_size=32,
      validation_data=(val_data, val_labels))

*Train on 1000 samples, validate on 100 samples
Epoch 1/10
1000/1000 [==============================] - 0s 124us/step - loss: 11.5267 - categorical_accuracy: 0.1070 - val_loss: 11.0015 - val_categorical_accuracy: 0.0500
Epoch 2/10
1000/1000 [==============================] - 0s 72us/step - loss: 11.5243 - categorical_accuracy: 0.0840 - val_loss: 10.9809 - val_categorical_accuracy: 0.1200
Epoch 3/10
1000/1000 [==============================] - 0s 73us/step - loss: 11.5213 - categorical_accuracy: 0.1000 - val_loss: 10.9945 - val_categorical_accuracy: 0.0800
Epoch 4/10
1000/1000 [==============================] - 0s 73us/step - loss: 11.5213 - categorical_accuracy: 0.1080 - val_loss: 10.9967 - val_categorical_accuracy: 0.0700
Epoch 5/10
1000/1000 [==============================] - 0s 73us/step - loss: 11.5181 - categorical_accuracy: 0.1150 - val_loss: 11.0184 - val_categorical_accuracy: 0.0500
Epoch 6/10
1000/1000 [==============================] - 0s 72us/step - loss: 11.5177 - categorical_accuracy: 0.1150 - val_loss: 10.9892 - val_categorical_accuracy: 0.0200
Epoch 7/10
1000/1000 [==============================] - 0s 72us/step - loss: 11.5130 - categorical_accuracy: 0.1320 - val_loss: 11.0038 - val_categorical_accuracy: 0.0500
Epoch 8/10
1000/1000 [==============================] - 0s 74us/step - loss: 11.5123 - categorical_accuracy: 0.1130 - val_loss: 11.0065 - val_categorical_accuracy: 0.0100
Epoch 9/10
1000/1000 [==============================] - 0s 72us/step - loss: 11.5076 - categorical_accuracy: 0.1150 - val_loss: 11.0062 - val_categorical_accuracy: 0.0800
Epoch 10/10
1000/1000 [==============================] - 0s 67us/step - loss: 11.5035 - categorical_accuracy: 0.1390 - val_loss: 11.0241 - val_categorical_accuracy: 0.1100*

### Input tf.data datasets ###

使用Datasets API可扩展到大型数据集或多设备训练。 将tf.data.Dataset实例传递给fit方法：

    # Instantiates a toy dataset instance:
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    
    # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
    model.fit(dataset, epochs=10, steps_per_epoch=30)

*Epoch 1/10
30/30 [==============================] - 0s 6ms/step - loss: 11.4973 - categorical_accuracy: 0.1406
Epoch 2/10
30/30 [==============================] - 0s 2ms/step - loss: 11.5182 - categorical_accuracy: 0.1344
Epoch 3/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4953 - categorical_accuracy: 0.1344
Epoch 4/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4842 - categorical_accuracy: 0.1542
Epoch 5/10
30/30 [==============================] - 0s 2ms/step - loss: 11.5081 - categorical_accuracy: 0.1510
Epoch 6/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4939 - categorical_accuracy: 0.1615
Epoch 7/10
30/30 [==============================] - 0s 2ms/step - loss: 11.5049 - categorical_accuracy: 0.1823
Epoch 8/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4617 - categorical_accuracy: 0.1760
Epoch 9/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4863 - categorical_accuracy: 0.1688
Epoch 10/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4946 - categorical_accuracy: 0.1885*

这里，fit方法使用steps_per_epoch参数 - 这是模型在移动到下一个周期之前运行的训练步数。 由于数据集生成批量数据，因此此代码段不需要batch_size。

数据集也可用于验证：

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(32).repeat()
    
    model.fit(dataset, epochs=10, steps_per_epoch=30,
      validation_data=val_dataset,
      validation_steps=3)

*Epoch 1/10
30/30 [==============================] - 0s 8ms/step - loss: 11.4649 - categorical_accuracy: 0.1740 - val_loss: 11.0269 - val_categorical_accuracy: 0.0521
Epoch 2/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4794 - categorical_accuracy: 0.1865 - val_loss: 11.4233 - val_categorical_accuracy: 0.0521
Epoch 3/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4604 - categorical_accuracy: 0.1760 - val_loss: 11.4040 - val_categorical_accuracy: 0.0208
Epoch 4/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4475 - categorical_accuracy: 0.1771 - val_loss: 11.3095 - val_categorical_accuracy: 0.2396
Epoch 5/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4727 - categorical_accuracy: 0.1750 - val_loss: 11.0481 - val_categorical_accuracy: 0.0938
Epoch 6/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4569 - categorical_accuracy: 0.1833 - val_loss: 11.3550 - val_categorical_accuracy: 0.1562
Epoch 7/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4653 - categorical_accuracy: 0.1958 - val_loss: 11.4325 - val_categorical_accuracy: 0.0417
Epoch 8/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4246 - categorical_accuracy: 0.1823 - val_loss: 11.3625 - val_categorical_accuracy: 0.0417
Epoch 9/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4542 - categorical_accuracy: 0.1729 - val_loss: 11.0326 - val_categorical_accuracy: 0.0521
Epoch 10/10
30/30 [==============================] - 0s 2ms/step - loss: 11.4600 - categorical_accuracy: 0.1979 - val_loss: 11.3494 - val_categorical_accuracy: 0.1042
*

### Evaluate and predict ###










