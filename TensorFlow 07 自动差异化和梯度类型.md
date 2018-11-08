自动差异化和梯度类型

[原文地址](https://tensorflow.google.cn/tutorials/eager/automatic_differentiation)

在上一个教程中，我们介绍了它们的张量和操作。 在本教程中，我们将介绍自动差异化，这是优化机器学习模型的关键技术。

## 安装 ##

    import tensorflow as tf
    tf.enable_eager_execution()
    
    tfe = tf.contrib.eager # Shorthand for some symbols

### 函数的衍生 ###

TensorFlow提供用于自动差异化的API - 计算函数的派生。 更接近模仿数学的方法是将计算封装在Python函数中，比如f，并使用tfe.gradients_function创建一个函数，该函数根据参数计算f的导数。 如果您熟悉autograd以区分numpy函数，那么这将是熟悉的。 例如：

    from math import pi
    
    def f(x):
      return tf.square(tf.sin(x))
    
    assert f(pi/2).numpy() == 1.0
    
    
    # grad_f will return a list of derivatives of f
    # with respect to its arguments. Since f() has a single argument,
    # grad_f will return a list with a single element.
    grad_f = tfe.gradients_function(f)
    assert tf.abs(grad_f(pi/2)[0]).numpy() < 1e-7

### 高阶梯度 ###

如果你喜欢，可以多次使用差异化API：

    def f(x):
      return tf.square(tf.sin(x))
    
    def grad(f):
      return lambda x: tfe.gradients_function(f)(x)[0]
    
    x = tf.lin_space(-2*pi, 2*pi, 100)  # 100 points between -2π and +2π
    
    import matplotlib.pyplot as plt
    
    plt.plot(x, f(x), label="f")
    plt.plot(x, grad(f)(x), label="first derivative")
    plt.plot(x, grad(grad(f))(x), label="second derivative")
    plt.plot(x, grad(grad(grad(f)))(x), label="third derivative")
    plt.legend()
    plt.show()

## Gradient tapes ##

每个可区分的TensorFlow操作都具有相关的梯度函数。 例如，tf.square（x）的梯度函数将是一个返回2.0 * x的函数。 要计算用户定义函数的梯度（如上例中的f（x）），TensorFlow首先“记录”应用于计算函数输出的所有操作。 我们将此记录称为“磁带”。 然后，它使用该磁带和与每个基元操作相关联的梯度函数，以使用反向模式区分来计算用户定义函数的梯度。

由于操作是在执行时记录的，因此自然会处理Python控制流程（例如使用ifs和whiles）：

    def f(x, y):
      output = 1
      # Must use range(int(y)) instead of range(y) in Python 3 when
      # using TensorFlow 1.10 and earlier. Can use range(y) in 1.11+
      for i in range(int(y)):
    	output = tf.multiply(output, x)
      return output
    
    def g(x, y):
      # Return the gradient of `f` with respect to it's first parameter
      return tfe.gradients_function(f)(x, y)[0]
    
    assert f(3.0, 2).numpy() == 9.0   # f(x, 2) is essentially x * x
    assert g(3.0, 2).numpy() == 6.0   # And its gradient will be 2 * x
    assert f(4.0, 3).numpy() == 64.0  # f(x, 3) is essentially x * x * x
    assert g(4.0, 3).numpy() == 48.0  # And its gradient will be 3 * x * x

有时，将感兴趣的计算封装到函数中可能是不方便的。 例如，如果希望输出的梯度相对于函数中计算的中间值。 在这种情况下，稍微更详细但明确的tf.GradientTape上下文是有用的。 tf.GradientTape上下文中的所有计算都被“记录”。

例如：

    x = tf.ones((2, 2))
      
    # a single t.gradient() call when the bug is resolved.
    with tf.GradientTape(persistent=True) as t:
      t.watch(x)
      y = tf.reduce_sum(x)
      z = tf.multiply(y, y)
    
    # Use the same tape to compute the derivative of z with respect to the
    # intermediate value y.
    dz_dy = t.gradient(z, y)
    assert dz_dy.numpy() == 8.0
    
    # Derivative of z with respect to the original input tensor x
    dz_dx = t.gradient(z, x)
    for i in [0, 1]:
      for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0

### 高阶梯度 ###

记录GradientTape上下文管理器内部的操作以实现自动差异化。如果在该上下文中计算梯度，则也记录梯度计算。因此，完全相同的API也适用于高阶梯度。例如：

    x = tf.constant(1.0)  # Convert the Python 1.0 to a Tensor object
    
    with tf.GradientTape() as t:
      with tf.GradientTape() as t2:
    	t2.watch(x)
    	y = x * x * x
      # Compute the gradient inside the 't' context manager
      # which means the gradient computation is differentiable as well.
      dy_dx = t2.gradient(y, x)
    d2y_dx2 = t.gradient(dy_dx, x)
    
    assert dy_dx.numpy() == 3.0
    assert d2y_dx2.numpy() == 6.0

## 接下来 ##

在本教程中，我们介绍了TensorFlow中的梯度计算。 有了这个，我们就拥有了构建和训练神经网络所需的足够原语。