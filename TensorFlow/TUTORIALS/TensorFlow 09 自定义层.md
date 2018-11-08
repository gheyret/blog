自定义层Layers

[原文地址](https://tensorflow.google.cn/tutorials/eager/custom_layers)

我们建议使用tf.keras作为构建神经网络的高级API。 也就是说，大多数TensorFlow API都可以在eager execution使用。

    import tensorflow as tf
    tfe = tf.contrib.eager
    
    tf.enable_eager_execution()

## Layers:常用的操作集 ##

大多数情况下，在为机器学习模型编写代码时，您希望在比单个操作和单个变量操作更高的抽象级别上操作。

许多机器学习模型表达的组成和相对简单的图层叠加，TensorFlow提供了一组常见的层以及简单的方法为你写你自己的应用程序特定的层从头开始或作为现有层组成。

TensorFlow在tf.keras包中包含完整的Keras API，而Keras层在构建自己的模型时非常有用。

    # In the tf.keras.layers package, layers are objects. To construct a layer,
    # simply construct the object. Most layers take as a first argument the number
    # of output dimensions / channels.
    layer = tf.keras.layers.Dense(100)
    # The number of input dimensions is often unnecessary, as it can be inferred
    # the first time the layer is used, but it can be provided if you want to 
    # specify it manually, which is useful in some complex models.
    layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

可以在文档中看到预先存在的图层的完整列表。 它包括Dense（完全连接层），Conv2D，LSTM，BatchNormalization，Dropout等等。

    # To use a layer, simply call it.
    layer(tf.zeros([10, 5]))

    # Layers have many useful methods. For example, you can inspect all variables
    # in a layer by calling layer.variables. In this case a fully-connected layer
    # will have variables for weights and biases.
    layer.variables

*[,
 ]*

    # The variables are also accessible through nice accessors
    layer.kernel, layer.bias

*(,
 )*

## Implementing custom layers 实现自定义图层 ##

实现自己的层的最佳方法是扩展tf.keras.Layer类并实现：* __init__，您可以在其中执行所有与输入无关的初始化*构建，您可以在其中了解输入张量的形状，并可以执行其余的 初始化*调用，在那里进行正向计算

请注意，您不必等到调用build来创建变量，您也可以在__init__中创建它们。 但是，在构建中创建它们的优点是它可以根据图层将要操作的输入的形状启用后期变量创建。 另一方面，在__init__中创建变量意味着需要明确指定创建变量所需的形状。

    class MyDenseLayer(tf.keras.layers.Layer):
      def __init__(self, num_outputs):
	    super(MyDenseLayer, self).__init__()
	    self.num_outputs = num_outputs
    
      def build(self, input_shape):
	    self.kernel = self.add_variable("kernel", shape=[input_shape[-1].value, self.num_outputs])
    
      def call(self, input):
    	return tf.matmul(input, self.kernel)
      
    layer = MyDenseLayer(10)
    print(layer(tf.zeros([10, 5])))
    print(layer.variables)

*tf.Tensor(
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape=(10, 10), dtype=float32)
[<tf.Variable 'my_dense_layer/kernel:0' shape=(5, 10) dtype=float32, numpy=
array([[-0.3605492 , -0.04739225, -0.02951819, -0.18543416, -0.44332704,
        -0.0950976 ,  0.28126985, -0.0250963 ,  0.32212436, -0.5181693 ],
       [-0.21757367,  0.526064  ,  0.06123459,  0.1903969 ,  0.05790561,
        -0.5072588 , -0.21904686,  0.5173026 ,  0.43780798,  0.45725197],
       [ 0.50270194, -0.27167848, -0.05042917,  0.46270257, -0.26882282,
        -0.4947695 ,  0.16789073, -0.15154582, -0.14892238,  0.31738114],
       [ 0.53172463,  0.57945853, -0.56659925, -0.18533885, -0.41237652,
        -0.5501916 , -0.44297028, -0.62930524,  0.0861659 , -0.30785066],
       [ 0.37747675,  0.4844882 , -0.17520931, -0.26652238, -0.61931545,
        -0.5237198 , -0.15267053,  0.44191295,  0.17881542, -0.4521292 ]],
      dtype=float32)>]*

请注意，您不必等到调用build来创建变量，您也可以在__init__中创建它们。

如果尽可能使用标准层，则整体代码更易于阅读和维护，因为其他读者将熟悉标准层的行为。 如果你想使用tf.keras.layers或tf.contrib.layers中不存在的图层，请考虑提交github问题，或者更好的是，向我们发送拉取请求！

## 模型：构建层 Models: composing layers ##

机器学习模型中许多有趣的层状事物是通过组合现有层来实现的。 例如，resnet中的每个残余块是卷积，批量标准化和快捷方式的组合。

创建包含其他图层的类似图层的东西时使用的主类是tf.keras.Model。 实现一个是通过继承自tf.keras.Model完成的。

    class ResnetIdentityBlock(tf.keras.Model):
      def __init__(self, kernel_size, filters):
	    super(ResnetIdentityBlock, self).__init__(name='')
	    filters1, filters2, filters3 = filters
	    
	    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
	    self.bn2a = tf.keras.layers.BatchNormalization()
	    
	    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
	    self.bn2b = tf.keras.layers.BatchNormalization()
	    
	    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
	    self.bn2c = tf.keras.layers.BatchNormalization()
    
      def call(self, input_tensor, training=False):
	    x = self.conv2a(input_tensor)
	    x = self.bn2a(x, training=training)
	    x = tf.nn.relu(x)
	    
	    x = self.conv2b(x)
	    x = self.bn2b(x, training=training)
	    x = tf.nn.relu(x)
	    
	    x = self.conv2c(x)
	    x = self.bn2c(x, training=training)
	    
	    x += input_tensor
	    return tf.nn.relu(x)
    
    
    block = ResnetIdentityBlock(1, [1, 2, 3])
    print(block(tf.zeros([1, 2, 3, 3])))
    print([x.name for x in block.variables])

*tf.Tensor(
[[[[0. 0. 0.]
   [0. 0. 0.]
   [0. 0. 0.]]

  [[0. 0. 0.]
   [0. 0. 0.]
   [0. 0. 0.]]]], shape=(1, 2, 3, 3), dtype=float32)
['resnet_identity_block/conv2d/kernel:0', 'resnet_identity_block/conv2d/bias:0', 'resnet_identity_block/batch_normalization/gamma:0', 'resnet_identity_block/batch_normalization/beta:0', 'resnet_identity_block/conv2d_1/kernel:0', 'resnet_identity_block/conv2d_1/bias:0', 'resnet_identity_block/batch_normalization_1/gamma:0', 'resnet_identity_block/batch_normalization_1/beta:0', 'resnet_identity_block/conv2d_2/kernel:0', 'resnet_identity_block/conv2d_2/bias:0', 'resnet_identity_block/batch_normalization_2/gamma:0', 'resnet_identity_block/batch_normalization_2/beta:0', 'resnet_identity_block/batch_normalization/moving_mean:0', 'resnet_identity_block/batch_normalization/moving_variance:0', 'resnet_identity_block/batch_normalization_1/moving_mean:0', 'resnet_identity_block/batch_normalization_1/moving_variance:0', 'resnet_identity_block/batch_normalization_2/moving_mean:0', 'resnet_identity_block/batch_normalization_2/moving_variance:0']*

然而，在很多时候，组成许多层的模型只是将一层接一层地称为一层。 这可以使用tf.keras.Sequential在非常少的代码中完成。

     my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
	       tf.keras.layers.BatchNormalization(),
	       tf.keras.layers.Conv2D(2, 1, 
	      padding='same'),
	       tf.keras.layers.BatchNormalization(),
	       tf.keras.layers.Conv2D(3, (1, 1)),
	       tf.keras.layers.BatchNormalization()])

    my_seq(tf.zeros([1, 2, 3, 3]))

## 下一步 ##

Now you can go back to the previous notebook and adapt the linear regression example to use layers and models to be better structured.

现在，您可以回到之前的笔记本并调整线性回归示例，以使用更好的结构化图层和模型。













