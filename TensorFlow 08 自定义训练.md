自定义训练：基础知识

[原文地址](https://tensorflow.google.cn/tutorials/eager/custom_training)

在上一个教程中，我们介绍了用于自动区分的TensorFlow API，这是机器学习的基本构建块。 在本教程中，我们将使用先前教程中介绍的TensorFlow原语来进行一些简单的机器学习。

TensorFlow还包括一个更高级别的神经网络API（tf.keras），它提供了有用的抽象来减少样板。 我们强烈建议那些使用神经网络的人使用更高级别的API。 但是，在这个简短的教程中，我们从第一原理开始介绍神经网络训练，以建立坚实的基础。

## 安装 ##

    import tensorflow as tf
    tf.enable_eager_execution()

## 变量 ##

TensorFlow中的张量是不可变的无状态对象。 然而，机器学习模型需要具有变化状态：随着模型训练，计算预测的相同代码应该随着时间的推移而表现不同（希望具有较低的损失！）。 要表示需要在计算过程中进行更改的状态，您可以选择依赖Python是一种有状态编程语言的事实：

    # Using python state
    x = tf.zeros([10, 10])
    x += 2  # This is equivalent to x = x + 2, which does not mutate the original value of x
    print(x)

*tf.Tensor(
[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]], shape=(10, 10), dtype=float32)*

但是，TensorFlow内置了有状态操作，这些通常比您的状态的低级Python表示更易于使用。 例如，为了表示模型中的权重，使用TensorFlow变量通常是方便有效的。

变量是一个存储值的对象，当在TensorFlow计算中使用时，它将隐式地从该存储值中读取。 有操作（tf.assign_sub，tf.scatter_update等）操作存储在TensorFlow变量中的值。

    v = tf.Variable(1.0)
    assert v.numpy() == 1.0
    
    # Re-assign the value
    v.assign(3.0)
    assert v.numpy() == 3.0
    
    # Use `v` in a TensorFlow operation like tf.square() and reassign
    v.assign(tf.square(v))
    assert v.numpy() == 9.0

在计算梯度时，会自动跟踪使用变量的计算。 对于表示嵌入的变量，TensorFlow默认会进行稀疏更新，这样可以提高计算效率和内存效率。

使用变量也是一种快速让代码的读者知道这段状态是可变的方法。

## 示例：拟合线性模型 ##

现在让我们把我们迄今为止的几个概念--- Tensor，GradientTape，Variable ---构建并训练一个简单的模型。 这通常涉及几个步骤：

1. 定义模型。
2. 定义损失函数。
3. 获取训练数据。
4. 运行训练数据并使用“优化器”调整变量以拟合数据。  

在本教程中，我们将介绍简单线性模型的一个简单示例：f（x）= x * W + b，它有两个变量 - W和b。 此外，我们将合成数据，使训练有素的模型具有W = 3.0和b = 2.0。

### 定义模型 ###

让我们定义一个简单的类来封装变量和计算。

    class Model(object):
      def __init__(self):
	    # Initialize variable to (5.0, 0.0)
	    # In practice, these should be initialized to random values.
	    self.W = tf.Variable(5.0)
	    self.b = tf.Variable(0.0)
    
      def __call__(self, x):
    	return self.W * x + self.b
      
    model = Model()    
    assert model(3.0).numpy() == 15.0

### 定义损失函数 ###

损失函数测量给定输入的模型输出与期望输出的匹配程度。 让我们使用标准的L2损失。

    def loss(predicted_y, desired_y):
      return tf.reduce_mean(tf.square(predicted_y - desired_y))

### 获取训练数据 ###

让我们用一些噪点合成训练数据。

    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    
    inputs  = tf.random_normal(shape=[NUM_EXAMPLES])
    noise   = tf.random_normal(shape=[NUM_EXAMPLES])
    outputs = inputs * TRUE_W + TRUE_b + noise

在我们训练模型之前，让我们可视化模型现在的位置。 我们将用红色绘制模型的预测，用蓝色绘制训练数据。

    import matplotlib.pyplot as plt
    
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.show()
    
    print('Current loss: '),
    print(loss(model(inputs), outputs).numpy())

*Current loss: 
9.285378*

### 定义训练循环 ###

我们现在拥有我们的网络和训练数据。 让我们训练它，即使用训练数据来更新模型的变量（W和b），以便使用梯度下降来减少损失。 在tf.train.Optimizer实现中捕获了许多梯度下降方案的变体。 我们强烈建议使用这些实现，但本着从第一原则构建的精神，在这个特定的例子中，我们将自己实现基本的数学。

    def train(model, inputs, outputs, learning_rate):
      with tf.GradientTape() as t:
    	current_loss = loss(model(inputs), outputs)
      dW, db = t.gradient(current_loss, [model.W, model.b])
      model.W.assign_sub(learning_rate * dW)
      model.b.assign_sub(learning_rate * db)

最后，让我们反复浏览训练数据，看看W和b是如何演变的。

    model = Model()
    
    # Collect the history of W-values and b-values to plot later
    Ws, bs = [], []
    epochs = range(10)
    for epoch in epochs:
      Ws.append(model.W.numpy())
      bs.append(model.b.numpy())
      current_loss = loss(model(inputs), outputs)
    
      train(model, inputs, outputs, learning_rate=0.1)
      print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch, Ws[-1], bs[-1], current_loss))
    
    # Let's plot it all
    plt.plot(epochs, Ws, 'r', epochs, bs, 'b')
    plt.plot([TRUE_W] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
    plt.legend(['W', 'b', 'true W', 'true_b'])
    plt.show()
      
*Epoch  0: W=5.00 b=0.00, loss=9.28538
Epoch  1: W=4.59 b=0.42, loss=6.21388
Epoch  2: W=4.27 b=0.75, loss=4.29427
Epoch  3: W=4.01 b=1.01, loss=3.09455
Epoch  4: W=3.81 b=1.22, loss=2.34476
Epoch  5: W=3.65 b=1.38, loss=1.87615
Epoch  6: W=3.53 b=1.51, loss=1.58328
Epoch  7: W=3.43 b=1.62, loss=1.40025
Epoch  8: W=3.35 b=1.70, loss=1.28586
Epoch  9: W=3.28 b=1.76, loss=1.21436*

![](https://i.imgur.com/JlEIyLC.png)

## 接下来 ##

在本教程中，我们介绍了变量，并使用到目前为止讨论的TensorFlow原语构建和训练了一个简单的线性模型。

从理论上讲，这几乎是您使用TensorFlow进行机器学习研究所需要的全部内容。 在实践中，特别是对于神经网络，像tf.keras这样的高级API将更加方便，因为它提供了更高级别的构建块（称为“层”），用于保存和恢复状态的实用程序，一套损失函数，套件 优化策略等。