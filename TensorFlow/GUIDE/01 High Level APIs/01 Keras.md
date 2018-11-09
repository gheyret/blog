
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

tf.keras.Model.evaluate和tf.keras.Model.predict方法可以使用NumPy数据和tf.data.Dataset。

要评估所提供数据的推理模式损失和指标：

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    
    model.evaluate(data, labels, batch_size=32)
    
    model.evaluate(dataset, steps=30)
    
*1000/1000 [==============================] - 0s 83us/step
30/30 [==============================] - 0s 3ms/step*

*[11.43181880315145, 0.18333333333333332]*

    result = model.predict(data, batch_size=32)
    print(result.shape)

*(1000, 10)*

## Build advanced models ##

### Functional API ###

tf.keras.Sequential模型是一个简单的图层堆栈，不能代表任意模型。 使用Keras功能API构建复杂的模型拓扑，例如：

     Multi-input models多输入型号，  
     Multi-output models多输出型号，  
     具有共享层的模型（同一层被调用多次），  
     具有非顺序数据流的模型（例如，残余连接）。

使用功能API构建模型的工作方式如下：

     1. 图层实例可调用并返回张量。  
     2. 输入张量和输出张量用于定义tf.keras.Model实例。  
     3. 这个模型的训练就像顺序模型一样。

以下示例使用功能API构建一个简单，完全连接的网络：

    inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor
    
    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)

实例化给定输入和输出的模型。

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    
    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
    
    # Trains for 5 epochs
    model.fit(data, labels, batch_size=32, epochs=5)

*Epoch 1/5
1000/1000 [==============================] - 0s 260us/step - loss: 11.7190 - acc: 0.1080
Epoch 2/5
1000/1000 [==============================] - 0s 75us/step - loss: 11.5347 - acc: 0.1010
Epoch 3/5
1000/1000 [==============================] - 0s 74us/step - loss: 11.5020 - acc: 0.1100
Epoch 4/5
1000/1000 [==============================] - 0s 75us/step - loss: 11.4908 - acc: 0.1090
Epoch 5/5
1000/1000 [==============================] - 0s 74us/step - loss: 11.4809 - acc: 0.1330
*

### Model subclassing ###

通过继承tf.keras.Model并定义自己的前向传递来构建完全可自定义的模型。 在__init__方法中创建图层并将它们设置为类实例的属性。 在call方法中定义前向传递。

当启用eager执行时，模型子类化特别有用，因为可以强制写入前向传递。

*关键点：为工作使用正确的API。 虽然模型子类化提供了灵活性，但其代价是更高的复杂性和更多的用户错误机会。 如果可能，请选择功能API。*

以下示例显示了使用自定义正向传递的子类tf.keras.Model：

    class MyModel(tf.keras.Model):
    
      def __init__(self, num_classes=10):
	    super(MyModel, self).__init__(name='my_model')
	    self.num_classes = num_classes
	    # Define your layers here.
	    self.dense_1 = layers.Dense(32, activation='relu')
	    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')
    
      def call(self, inputs):
	    # Define your forward pass here,
	    # using layers you previously defined (in `__init__`).
	    x = self.dense_1(inputs)
	    return self.dense_2(x)
    
      def compute_output_shape(self, input_shape):
	    # You need to override this function if you want to use the subclassed model
	    # as part of a functional-style model.
	    # Otherwise, this method is optional.
	    shape = tf.TensorShape(input_shape).as_list()
	    shape[-1] = self.num_classes
	    return tf.TensorShape(shape)

实例化新的模型类：

    model = MyModel(num_classes=10)
    
    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
    
    # Trains for 5 epochs.
    model.fit(data, labels, batch_size=32, epochs=5)

*Epoch 1/5
1000/1000 [==============================] - 0s 224us/step - loss: 11.5206 - acc: 0.0990
Epoch 2/5
1000/1000 [==============================] - 0s 62us/step - loss: 11.5128 - acc: 0.1070
Epoch 3/5
1000/1000 [==============================] - 0s 64us/step - loss: 11.5023 - acc: 0.0980
Epoch 4/5
1000/1000 [==============================] - 0s 65us/step - loss: 11.4941 - acc: 0.0980
Epoch 5/5
1000/1000 [==============================] - 0s 66us/step - loss: 11.4879 - acc: 0.0990
*

### Custom layers ###

通过继承tf.keras.layers.Layer并实现以下方法来创建自定义层：

1. build：创建图层的权重。 使用add_weight方法添加权重。  
2. call：定义前向传播。  
3. compute_output_shape：指定在给定输入形状的情况下如何计算图层的输出形状。  
4. Optionally，可以通过实现get_config方法和from_config类方法来序列化层。

这是一个自定义层的示例，它使用内核矩阵实现输入的matmul：

    class MyLayer(layers.Layer):
    
      def __init__(self, output_dim, **kwargs):
	    self.output_dim = output_dim
	    super(MyLayer, self).__init__(**kwargs)
    
      def build(self, input_shape):
	    shape = tf.TensorShape((input_shape[1], self.output_dim))
	    # Create a trainable weight variable for this layer.
	    self.kernel = self.add_weight(name='kernel',
	      shape=shape,
	      initializer='uniform',
	      trainable=True)
	    # Be sure to call this at the end
	    super(MyLayer, self).build(input_shape)
	    
      def call(self, inputs):
    	return tf.matmul(inputs, self.kernel)
    
      def compute_output_shape(self, input_shape):
	    shape = tf.TensorShape(input_shape).as_list()
	    shape[-1] = self.output_dim
	    return tf.TensorShape(shape)
	    
      def get_config(self):
	    base_config = super(MyLayer, self).get_config()
	    base_config['output_dim'] = self.output_dim
	    return base_config
	    
      @classmethod
      def from_config(cls, config):
    	return cls(**config)

使用自定义图层创建模型：

    model = tf.keras.Sequential([
	    MyLayer(10),
	    layers.Activation('softmax')])
    
    # The compile step specifies the training configuration
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
    
    # Trains for 5 epochs.
    model.fit(data, labels, batch_size=32, epochs=5)

*Epoch 1/5
1000/1000 [==============================] - 0s 170us/step - loss: 11.4872 - acc: 0.0990
Epoch 2/5
1000/1000 [==============================] - 0s 52us/step - loss: 11.4817 - acc: 0.0910
Epoch 3/5
1000/1000 [==============================] - 0s 52us/step - loss: 11.4800 - acc: 0.0960
Epoch 4/5
1000/1000 [==============================] - 0s 57us/step - loss: 11.4778 - acc: 0.0960
Epoch 5/5
1000/1000 [==============================] - 0s 60us/step - loss: 11.4764 - acc: 0.0930*

## Callbacks ##

回调是传递给模型的对象，用于在训练期间自定义和扩展其行为。 您可以编写自己的自定义回调，或使用包含以下内置的tf.keras.callbacks：

tf.keras.callbacks.ModelCheckpoint：定期保存模型的检查点。  
tf.keras.callbacks.LearningRateScheduler：动态改变学习率。  
tf.keras.callbacks.EarlyStopping：验证性能停止改进时的中断培训。  
tf.keras.callbacks.TensorBoard：使用TensorBoard监控模型的行为。

要使用tf.keras.callbacks.Callback，请将其传递给模型的fit方法：

    callbacks = [
      # Interrupt training if `val_loss` stops improving for over 2 epochs
      tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
      # Write TensorBoard logs to `./logs` directory
      tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
      validation_data=(val_data, val_labels))

*Train on 1000 samples, validate on 100 samples
Epoch 1/5
1000/1000 [==============================] - 0s 150us/step - loss: 11.4748 - acc: 0.1230 - val_loss: 10.9787 - val_acc: 0.1000
Epoch 2/5
1000/1000 [==============================] - 0s 78us/step - loss: 11.4730 - acc: 0.1060 - val_loss: 10.9783 - val_acc: 0.1300
Epoch 3/5
1000/1000 [==============================] - 0s 82us/step - loss: 11.4711 - acc: 0.1130 - val_loss: 10.9756 - val_acc: 0.1500
Epoch 4/5
1000/1000 [==============================] - 0s 82us/step - loss: 11.4704 - acc: 0.1050 - val_loss: 10.9772 - val_acc: 0.0900
Epoch 5/5
1000/1000 [==============================] - 0s 83us/step - loss: 11.4689 - acc: 0.1140 - val_loss: 10.9781 - val_acc: 0.1300
*

## Save and restore ##

### Weights only ###

使用tf.keras.Model.save_weights保存并加载模型的权重：

    model = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')])
    
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])

    # Save weights to a TensorFlow Checkpoint file
    model.save_weights('./weights/my_model')
    
    # Restore the model's state,
    # this requires a model with the same architecture.
    model.load_weights('./weights/my_model')

默认情况下，这会以TensorFlow检查点文件格式保存模型的权重。 权重也可以保存为Keras HDF5格式（Keras的多后端实现的默认值）：

    # Save weights to a HDF5 file
    model.save_weights('my_model.h5', save_format='h5')
    
    # Restore the model's state
    model.load_weights('my_model.h5')

### Configuration only ###

可以保存模型的配置 - 这可以在没有任何权重的情况下序列化模型体系结构。 即使没有定义原始模型的代码，保存的配置也可以重新创建和初始化相同的模型。 Keras支持JSON和YAML序列化格式：

    # Serialize a model to JSON format
    json_string = model.to_json()
    json_string

*'{"backend": "tensorflow", "keras_version": "2.1.6-tf", "config": {"name": "sequential_3", "layers": [{"config": {"units": 64, "kernel_regularizer": null, "activation": "relu", "bias_constraint": null, "trainable": true, "use_bias": true, "bias_initializer": {"config": {"dtype": "float32"}, "class_name": "Zeros"}, "activity_regularizer": null, "dtype": null, "kernel_constraint": null, "kernel_initializer": {"config": {"mode": "fan_avg", "seed": null, "distribution": "uniform", "scale": 1.0, "dtype": "float32"}, "class_name": "VarianceScaling"}, "name": "dense_17", "bias_regularizer": null}, "class_name": "Dense"}, {"config": {"units": 10, "kernel_regularizer": null, "activation": "softmax", "bias_constraint": null, "trainable": true, "use_bias": true, "bias_initializer": {"config": {"dtype": "float32"}, "class_name": "Zeros"}, "activity_regularizer": null, "dtype": null, "kernel_constraint": null, "kernel_initializer": {"config": {"mode": "fan_avg", "seed": null, "distribution": "uniform", "scale": 1.0, "dtype": "float32"}, "class_name": "VarianceScaling"}, "name": "dense_18", "bias_regularizer": null}, "class_name": "Dense"}]}, "class_name": "Sequential"}'*

    import json
    import pprint
    pprint.pprint(json.loads(json_string))

{'backend': 'tensorflow',
 'class_name': 'Sequential',
 'config': {'layers': [{'class_name': 'Dense',
                        'config': {'activation': 'relu',
                                   *'activity_regularizer': None,
                                   'bias_constraint': None,
                                   'bias_initializer': {'class_name': 'Zeros',
                                                        'config': {'dtype': 'float32'}},
                                   'bias_regularizer': None,
                                   'dtype': None,
                                   'kernel_constraint': None,
                                   'kernel_initializer': {'class_name': 'VarianceScaling',
                                                          'config': {'distribution': 'uniform',
                                                                     'dtype': 'float32',
                                                                     'mode': 'fan_avg',
                                                                     'scale': 1.0,
                                                                     'seed': None}},
                                   'kernel_regularizer': None,
                                   'name': 'dense_17',
                                   'trainable': True,
                                   'units': 64,
                                   'use_bias': True}},
                       {'class_name': 'Dense',
                        'config': {'activation': 'softmax',
                                   'activity_regularizer': None,
                                   'bias_constraint': None,
                                   'bias_initializer': {'class_name': 'Zeros',
                                                        'config': {'dtype': 'float32'}},
                                   'bias_regularizer': None,
                                   'dtype': None,
                                   'kernel_constraint': None,
                                   'kernel_initializer': {'class_name': 'VarianceScaling',
                                                          'config': {'distribution': 'uniform',
                                                                     'dtype': 'float32',
                                                                     'mode': 'fan_avg',
                                                                     'scale': 1.0,
                                                                     'seed': None}},
                                   'kernel_regularizer': None,
                                   'name': 'dense_18',
                                   'trainable': True,
                                   'units': 10,
                                   'use_bias': True}}],
            'name': 'sequential_3'},
 'keras_version': '2.1.6-tf'}*

从json重新创建模型（刚刚初始化）。

    fresh_model = tf.keras.models.model_from_json(json_string)

将模型序列化为YAML格式

    yaml_string = model.to_yaml()
    print(yaml_string)

*backend: tensorflow
class_name: Sequential
config:
  layers:
  - class_name: Dense
    config:
      activation: relu
      activity_regularizer: null
      bias_constraint: null
      bias_initializer:
        class_name: Zeros
        config: {dtype: float32}
      bias_regularizer: null
      dtype: null
      kernel_constraint: null
      kernel_initializer:
        class_name: VarianceScaling
        config: {distribution: uniform, dtype: float32, mode: fan_avg, scale: 1.0,
          seed: null}
      kernel_regularizer: null
      name: dense_17
      trainable: true
      units: 64
      use_bias: true
  - class_name: Dense
    config:
      activation: softmax
      activity_regularizer: null
      bias_constraint: null
      bias_initializer:
        class_name: Zeros
        config: {dtype: float32}
      bias_regularizer: null
      dtype: null
      kernel_constraint: null
      kernel_initializer:
        class_name: VarianceScaling
        config: {distribution: uniform, dtype: float32, mode: fan_avg, scale: 1.0,
          seed: null}
      kernel_regularizer: null
      name: dense_18
      trainable: true
      units: 10
      use_bias: true
  name: sequential_3
keras_version: 2.1.6-tf*

从yaml重新创建模型

    fresh_model = tf.keras.models.model_from_yaml(yaml_string)

*注意：子类模型不可序列化，因为它们的体系结构由调用方法体中的Python代码定义。*

### Entire model ###

整个模型可以保存到包含权重值，模型配置甚至优化器配置的文件中。 这允许您检查模型并稍后从完全相同的状态恢复训练 - 无需访问原始代码。

    # Create a trivial model
    model = tf.keras.Sequential([
      layers.Dense(10, activation='softmax', input_shape=(32,)),
      layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='rmsprop',
      loss='categorical_crossentropy',
      metrics=['accuracy'])
    model.fit(data, labels, batch_size=32, epochs=5)
    
    
    # Save entire model to a HDF5 file
    model.save('my_model.h5')
    
    # Recreate the exact same model, including weights and optimizer.
    model = tf.keras.models.load_model('my_model.h5')

*Epoch 1/5
1000/1000 [==============================] - 0s 297us/step - loss: 11.5009 - acc: 0.0980
Epoch 2/5
1000/1000 [==============================] - 0s 76us/step - loss: 11.4844 - acc: 0.0960
Epoch 3/5
1000/1000 [==============================] - 0s 77us/step - loss: 11.4791 - acc: 0.0850
Epoch 4/5
1000/1000 [==============================] - 0s 78us/step - loss: 11.4771 - acc: 0.1020
Epoch 5/5
1000/1000 [==============================] - 0s 79us/step - loss: 11.4763 - acc: 0.0900*

## Eager execution ##

Eager execution是一个必要的编程环境，可以立即评估操作。 这对于Keras不是必需的，但是由tf.keras支持，对于检查程序和调试很有用。

所有tf.keras模型构建API都与Eager execution兼容。 虽然可以使用顺序和功能API，但是Eager execution尤其有利于模型子类化和构建自定义层 - 需要您将前向传递作为代码编写的API（而不是通过组合现有层来创建模型的API）。

有关使用具有自定义训练循环和tf.GradientTape的Keras模型的示例，请参阅Eager execution指南。

## Distribution ##

### Estimators ###

Estimators API用于分布式环境的训练模型。 这针对行业用例，例如可以导出模型进行生产的大型数据集的分布式训练。

通过使用tf.keras.estimator.model_to_estimator将模型转换为tf.estimator.Estimator对象，可以使用tf.estimator API训练tf.keras.Model。 请参阅从Keras模型创建Estimators。

    model = tf.keras.Sequential([layers.Dense(10,activation='softmax'),
      layers.Dense(10,activation='softmax')])
    
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
    
    estimator = tf.keras.estimator.model_to_estimator(model)

*INFO:tensorflow:Using the Keras model provided.
INFO:tensorflow:Using default config.
WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpm0ljzq8s
INFO:tensorflow:Using config: {'_experimental_distribute': None, '_master': '', '_eval_distribute': None, '_num_ps_replicas': 0, '_protocol': None, '_global_id_in_cluster': 0, '_save_summary_steps': 100, '_tf_random_seed': None, '_model_dir': '/tmp/tmpm0ljzq8s', '_evaluation_master': '', '_task_id': 0, '_keep_checkpoint_max': 5, '_save_checkpoints_steps': None, '_service': None, '_num_worker_replicas': 1, '_save_checkpoints_secs': 600, '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fad8c5d3e10>, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_train_distribute': None, '_task_type': 'worker', '_device_fn': None}*


*Note: Enable eager execution for debugging Estimator input functions and inspecting data.*

*注意：启用预先执行以调试Estimator输入函数和检查数据。*

### Multiple GPUs ###

tf.keras模型可以使用tf.contrib.distribute.DistributionStrategy在多个GPU上运行。 此API在多个GPU上提供分布式训练，几乎不对现有代码进行任何更改。

目前，tf.contrib.distribute.MirroredStrategy是唯一受支持的分发策略。 MirroredStrategy使用all-reduce在一台机器上进行同步训练的图形内复制。 要将DistributionStrategy与Keras一起使用，请使用tf.keras.estimator.model_to_estimator将tf.keras.Model转换为tf.estimator.Estimator，然后训练estimator

以下示例在单个计算机上的多个GPU之间分发tf.keras.Model。

首先，定义一个简单的模型：

    model = tf.keras.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.summary()

*_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_23 (Dense)             (None, 16)                176       
_________________________________________________________________
dense_24 (Dense)             (None, 1)                 17        
=================================================================
Total params: 193
Trainable params: 193
Non-trainable params: 0
_________________________________________________________________*

定义输入管道。 input_fn返回一个tf.data.Dataset对象，用于在多个设备之间分配数据 - 每个设备处理输入批处理的一部分。

    def input_fn():
      x = np.random.random((1024, 10))
      y = np.random.randint(2, size=(1024, 1))
      x = tf.cast(x, tf.float32)
      dataset = tf.data.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(10)
      dataset = dataset.batch(32)
      return dataset

接下来，创建一个tf.estimator.RunConfig并将train_distribute参数设置为tf.contrib.distribute.MirroredStrategy实例。 创建MirroredStrategy时，您可以指定设备列表或设置num_gpus参数。 默认使用所有可用的GPU，如下所示：

    strategy = tf.contrib.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)

*INFO:tensorflow:Initializing RunConfig with distribution strategies.
INFO:tensorflow:Not using Distribute Coordinator.*

将Keras模型转换为tf.estimator.Estimator实例：

    keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model,
      config=config,
      model_dir='/tmp/model_dir')

*INFO:tensorflow:Using the Keras model provided.
INFO:tensorflow:Using config: {'_experimental_distribute': None, '_master': '', '_eval_distribute': None, '_num_ps_replicas': 0, '_protocol': None, '_global_id_in_cluster': 0, '_save_summary_steps': 100, '_tf_random_seed': None, '_model_dir': '/tmp/model_dir', '_evaluation_master': '', '_task_id': 0, '_keep_checkpoint_max': 5, '_save_checkpoints_steps': None, '_service': None, '_num_worker_replicas': 1, '_save_checkpoints_secs': 600, '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7faed9e1c550>, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_distribute_coordinator_mode': None, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_train_distribute': <tensorflow.contrib.distribute.python.mirrored_strategy.MirroredStrategy object at 0x7faed9e1c588>, '_task_type': 'worker', '_device_fn': None}*

最后，通过提供input_fn和steps参数来训练Estimator实例：

    keras_estimator.train(input_fn=input_fn, steps=10)

*WARNING:tensorflow:Not all devices in DistributionStrategy are visible to TensorFlow session.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Warm-starting with WarmStartSettings: WarmStartSettings(ckpt_to_initialize_from='/tmp/model_dir/keras/keras_model.ckpt', vars_to_warm_start='.*', var_name_to_vocab_info={}, var_name_to_prev_var_name={})
INFO:tensorflow:Warm-starting from: ('/tmp/model_dir/keras/keras_model.ckpt',)
INFO:tensorflow:Warm-starting variable: dense_24/kernel; prev_var_name: Unchanged
INFO:tensorflow:Warm-starting variable: dense_23/bias; prev_var_name: Unchanged
INFO:tensorflow:Warm-starting variable: dense_24/bias; prev_var_name: Unchanged
INFO:tensorflow:Warm-starting variable: dense_23/kernel; prev_var_name: Unchanged
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into /tmp/model_dir/model.ckpt.
INFO:tensorflow:Initialize system
INFO:tensorflow:loss = 0.7582453, step = 0
INFO:tensorflow:Saving checkpoints for 10 into /tmp/model_dir/model.ckpt.
INFO:tensorflow:Finalize system.
INFO:tensorflow:Loss for final step: 0.6743419.*

## 结束 ##

