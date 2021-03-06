[原文地址](https://tensorflow.google.cn/tutorials/keras/overfit_and_underfit)

与往常一样，此示例中的代码将使用tf.keras API，您可以在TensorFlow Keras指南中了解更多信息。

在前面的两个例子中 - 分类电影评论和预测住房价格 - 我们看到我们的模型对验证数据的准确性在经过多个时期的训练后会达到峰值，然后开始下降。

换句话说，我们的模型会过度拟合训练数据。 学习如何处理过度拟合很重要。 尽管通常可以在训练集上实现高精度，但我们真正想要的是开发能够很好地泛化测试数据（或之前未见过的数据）的模型。

过度拟合的反面是欠拟合。 当测试数据仍有改进空间时，会发生欠拟合。 出现这种情况的原因有很多：如果模型不够强大，过度正则化，或者根本没有经过足够长时间的训练。 这意味着网络尚未学习训练数据中的相关模式。

如果训练时间过长，模型将开始过度拟合并从训练数据中学习模式，而这些模式不会泛化到测试数据。 我们需要取得平衡。 如下所述，了解如何训练适当数量的时期是一项有用的技能。

为了防止过度拟合，最好的解决方案是使用更多的训练数据。 受过更多数据训练的模型自然会更好地泛化。 当不再可能时，下一个最佳解决方案是使用正规化等技术。 这些限制了模型可以存储的信息的数量和类型。 如果一个网络只能记住少量的模式，那么优化过程将迫使它专注于最突出的模式，这些模式有更好的泛化性。

在本文中，我们将探索两种常见的正则化技术 - 权重正则化和丢失 - 并使用它们来改进我们的IMDB电影评论分类。

    import tensorflow as tf
    from tensorflow import keras
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print(tf.__version__)

*1.11.0*

## 下载IMDB数据集 ##

我们不会像以前的本章一样使用嵌入，而是对句子进行多热编码。 该模型将很快适应训练集。 它将用于证明何时发生过度拟合，以及如何对抗它。

对我们的列表进行多热编码意味着将它们转换为0和1的向量。 具体地说，这将意味着例如将序列[3,5]转换为10,000维向量，除了索引3和5之外，它将是全零，其将是1。

    NUM_WORDS = 10000
    
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    
    def multi_hot_sequences(sequences, dimension):
	    # Create an all-zero matrix of shape (len(sequences), dimension)
	    results = np.zeros((len(sequences), dimension))
	    for i, word_indices in enumerate(sequences):
	    results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
	    return results
    
    
    train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

让我们看一下生成的多热矢量。 单词索引按频率排序，因此预计索引零附近有更多的1值，我们可以在此图中看到：

    plt.plot(train_data[0])

![](https://i.imgur.com/Ble2yJl.png)

## 过拟合 ##

防止过度拟合的最简单方法是减小模型的大小，即模型中可学习参数的数量（由层数和每层单元数决定）。 在深度学习中，模型中可学习参数的数量通常被称为模型的“容量”。 直观地，具有更多参数的模型将具有更多“记忆能力”，因此将能够容易地学习训练样本与其目标之间的完美的字典式映射，没有任何泛化能力的映射，但是在做出预测时（新出现的数据）这将是无用的。

始终牢记这一点：深度学习模型往往善于拟合训练数据，但真正的挑战是泛化，而不是拟合。

另一方面，如果网络具有有限的记忆资源，则将不能容易地学习映射。 为了最大限度地减少损失，它必须学习具有更强预测能力的压缩表示。 同时，如果您使模型太小，则难以拟合训练数据。 “太多容量”和“容量不足”之间存在平衡。

不幸的是，没有神奇的公式来确定模型的正确尺寸或架构（就层数而言，或每层的正确尺寸）。 您将不得不尝试使用一系列不同的架构。

要找到合适的模型大小，最好从相对较少的图层和参数开始，然后开始增加图层的大小或添加新图层，直到您看到验证损失的收益递减为止。 让我们在我们的电影评论分类网络上试试这个。

我们将仅使用Dense图层作为基线创建一个简单模型，然后创建更小和更大的版本，并进行比较。

### 创建基线模型 ###

    baseline_model = keras.Sequential([
	    # `input_shape` is only required here so that `.summary` works. 
	    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(16, activation=tf.nn.relu),
	    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    baseline_model.compile(optimizer='adam',
       loss='binary_crossentropy',
       metrics=['accuracy', 'binary_crossentropy'])
    
    baseline_model.summary()

    baseline_history = baseline_model.fit(train_data,
	      train_labels,
	      epochs=20,
	      batch_size=512,
	      validation_data=(test_data, test_labels),
	      verbose=2)

### 创建较小的模型 ###

让我们创建一个隐藏单元较少的模型，与我们刚刚创建的基线模型进行比较：

    smaller_model = keras.Sequential([
	    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(4, activation=tf.nn.relu),
	    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    smaller_model.compile(optimizer='adam',
	    loss='binary_crossentropy',
	    metrics=['accuracy', 'binary_crossentropy'])
    
    smaller_model.summary()
    
并用相同的数据训练此模型：

    smaller_history = smaller_model.fit(train_data,
		    train_labels,
		    epochs=20,
		    batch_size=512,
		    validation_data=(test_data, test_labels),
		    verbose=2)

### 创建较大的模型 ###

作为练习，您可以创建一个更大的模型，并查看它开始过度拟合的速度。 接下来，让我们在这个基准测试中添加一个容量大得多的网络，远远超出问题的范围：

    bigger_model = keras.models.Sequential([
	    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(512, activation=tf.nn.relu),
	    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    bigger_model.compile(optimizer='adam',
	     loss='binary_crossentropy',
	     metrics=['accuracy','binary_crossentropy'])
    
    bigger_model.summary()

同样，再次用相同的数据训练模型：
    
    bigger_history = bigger_model.fit(train_data,train_labels,
	      epochs=20,
	      batch_size=512,
	      validation_data=(test_data, test_labels),
	      verbose=2)

### 绘制训练和验证损失 ###

实线表示训练损失，虚线表示验证损失（记住：较低的验证损失表示更好的模型）。 在这里，较小的网络开始过度拟合晚于基线模型（在6个时期之后而不是4个时期），并且一旦开始过度拟合，其性能下降得慢得多。

    def plot_history(histories, key='binary_crossentropy'):
      	plt.figure(figsize=(16,10))
    
      	for name, history in histories:
    		val = plt.plot(history.epoch, history.history['val_'+key],'--', label=name.title()+' Val')
    	plt.plot(history.epoch, history.history[key], color=val[0].get_color(),label=name.title()+' Train')
    
      	plt.xlabel('Epochs')
      	plt.ylabel(key.replace('_',' ').title())
      	plt.legend()
    
      	plt.xlim([0,max(history.epoch)])
    
    
    plot_history([('baseline', baseline_history),('smaller', smaller_history),('bigger', bigger_history)])

![](https://i.imgur.com/kYN3OQI.png)

请注意，较大的网络在仅仅一个时期之后几乎立即开始过度拟合，并且更加严重地过度配置。 网络容量越大，能够越快地对训练数据进行建模（导致训练损失低），但过度拟合的可能性越大（导致训练和验证损失之间的差异很大）。

## 策略 ##

### 权重正则化 ###

你可能熟悉奥卡姆的剃刀原则：给出两个解释的东西，最可能是正确的解释是“最简单”的解释，即做出最少量假设的解释。 这也适用于神经网络学习的模型：给定一些训练数据和网络架构，有多组权重值（多个模型）可以解释数据，而简单模型比复杂模型更不容易过度拟合。

在这种情况下，“简单模型”是一个模型，其中参数值的分布具有较少的熵（或者具有较少参数的模型，如我们在上面的部分中所见）。因此，减轻过度拟合的常见方法是通过强制其权重仅采用较小的值来对网络的复杂性施加约束，这使得权重值的分布更“规则”。这被称为“权重正则化”，并且通过向网络的损失函数添加与具有大权重相关联的成本来完成。这个成本有两种：

L1正则化，其中所添加的成本与权重系数的绝对值成比例（即，称为权重的“L1范数”）。

L2正则化，其中所添加的成本与权重系数的值的平方成比例（即，与权重的所谓“L2范数”）成比例。 L2正则化在神经网络的背景下也称为权重衰减。不要让不同的名字让你感到困惑：权重衰减在数学上与L2正则化完全相同。

在tf.keras中，通过将权重正则化实例作为关键字参数传递给层来添加权重正则化。 现在让我们添加L2权重正则化。

    l2_model = keras.models.Sequential([
	    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
	       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
	       activation=tf.nn.relu),
	    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    l2_model.compile(optimizer='adam',
	     loss='binary_crossentropy',
	     metrics=['accuracy', 'binary_crossentropy'])
    
    l2_model_history = l2_model.fit(train_data, train_labels,
	    epochs=20,
	    batch_size=512,
	    validation_data=(test_data, test_labels),
	    verbose=2)

l2（0.001）表示该层的权重矩阵中的每个系数将为网络的总损失增加0.001 * weight_coefficient_value ** 2。 请注意，由于此惩罚仅在训练时添加，因此在训练时此网络的损失将远高于在测试时间。

这是我们的L2正规化惩罚的影响：

    plot_history([('baseline', baseline_history),
      ('l2', l2_model_history)])

![](https://i.imgur.com/cG7jQgv.png)

正如您所看到的，L2正则化模型比基线模型更能抵抗过度拟合，即使两个模型具有相同数量的参数。

###Add Dropout ###

Dropout是由Hinton和他在多伦多大学的学生开发的最有效和最常用的神经网络正则化技术之一。应用于层的Dropout包括在训练期间随机“丢弃”（即设置为零）该层的多个输出特征。假设一个给定的层通常会在训练期间为给定的输入样本返回一个向量[0.2,0.5,1.3,0.8,1.1];在应用了丢失之后，该向量将具有随机分布的几个零条目，例如， [0,0.5,1.3,0,1.1]。 “dropout rate”是被淘汰的特征的一部分;它通常设置在0.2和0.5之间。在测试时，没有单位被剔除，而是将图层的输出值按比例缩小等于辍学率的因子，以便平衡更多单位活跃的事实而不是训练时间。

在tf.keras中，您可以通过Dropout图层在网络中引入dropout，该图层将在之前应用于图层的输出。

让我们在IMDB网络中添加两个Dropout图层，看看它们在减少过度拟合方面做得如何：

    dpt_model = keras.models.Sequential([
	    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
	    keras.layers.Dropout(0.5),
	    keras.layers.Dense(16, activation=tf.nn.relu),
	    keras.layers.Dropout(0.5),
	    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    dpt_model.compile(optimizer='adam',
	      loss='binary_crossentropy',
	      metrics=['accuracy','binary_crossentropy'])
    
    dpt_model_history = dpt_model.fit(train_data, train_labels,
	      epochs=20,
	      batch_size=512,
	      validation_data=(test_data, test_labels),
	      verbose=2)

    plot_history([('baseline', baseline_history),
      ('dropout', dpt_model_history)])

![](https://i.imgur.com/GHLsbhw.png)

添加dropout是对基线模型的明显改进。

回顾一下：这里是防止神经网络中过度拟合的最常见方法：

1. 获取更多培训数据。
2. 减少网络容量。
3. 添加重量正规化。
4. 添加dropout。
本文未涉及的两个重要方法是数据增强和批量标准化。

完整代码：

    import tensorflow as tf
    from tensorflow import keras
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print(tf.__version__)
    
    
    # Download the IMDB dataset
    NUM_WORDS = 10000
    (train_data,train_labels),(test_data,test_labels) = keras.datasets.imdb.load_data(num_words = NUM_WORDS)
    def multi_hot_sequences(sequences,dimension):
	    # Create an all-zero matrix of shape (len(sequences),dimension)
	    results = np.zeros((len(sequences),dimension))
	    for i,word_indices in enumerate(sequences):
	    results[i,word_indices] = 1.0  # set specific indices of results[i] to 1s
	    return results
    
    train_data = multi_hot_sequences(train_data,dimension=NUM_WORDS)
    test_data = multi_hot_sequences(test_data,dimension=NUM_WORDS)
    
    plt.plot(train_data[0])
    plt.show()
    
    
    # Create a baseline model
    baseline_model = keras.Sequential([
	    # 'input_shpae' is only required here so that '.summary' works.
	    keras.layers.Dense(16,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(16,activation=tf.nn.relu),
	    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    
    baseline_model.compile(optimizer='adam',
       loss='binary_crossentropy',
       metrics=['accuracy','binary_crossentropy'])
    baseline_model.summary()
    
    baseline_history = baseline_model.fit(train_data,
      train_labels,
      epochs=20,
      batch_size=512,
      validation_data =(test_data,test_labels),
      verbose=2)
    
    
    # Create a smaller model
    smaller_model = keras.Sequential([
	    keras.layers.Dense(4,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(4,activation=tf.nn.relu),
	    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    
    smaller_model.compile(optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy','binary_crossentropy'])
    
    smaller_model.summary()
    
    smaller_history = smaller_model.fit(train_data,
	    train_labels,
	    epochs=20,
	    batch_size=512,
	    validation_data=(test_data,test_labels),
	    verbose=2)
    
    
    # Create a bigger model
    bigger_model = keras.models.Sequential([
	    keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(512,activation=tf.nn.relu),
	    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    
    bigger_model.compile(optimizer='adam',
       loss='binary_crossentropy',
       metrics=['accuracy','binary_crossentropy'])
    
    bigger_model.summary()
    
    bigger_history = bigger_model.fit(train_data,
	    train_labels,
	    epochs=20,
	    batch_size=512,
	    validation_data=(test_data,test_labels),
	    verbose=2)
    
    
    # Plot the training and validation loss
    def plot_history(histories,key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    
    for name,history in histories:
	    val = plt.plot(history.epoch,history.history['val_' + key],
	       '--',label = name.title() + ' Val')
    plt.plot(history.epoch,history.history[key],color=val[0].get_color(),
		label=name.title()+' Train')
    
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('-',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()
    
    plot_history([('baseline',baseline_history),
     ('smaller',smaller_history),
     ('bigger',bigger_history)])
    
    
    
    # Add weight regulatization
    l2_model = keras.models.Sequential([
	    keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
	       activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
	       activation=tf.nn.relu),
	    keras.layers.Dense(1,activation=tf.nn.sigmoid)])
    
    l2_model.compile(optimizer='adam',
	     loss='binary_crossentropy',
	     metrics=['accuracy','binary_crossentropy'])
    
    l2_model_history = l2_model.fit(train_data,train_labels,
	    epochs=20,
	    batch_size=512,
	    validation_data=(test_data,test_labels),
	    verbose=2)
    plot_history([('baseline',baseline_history),
      	('l2',l2_model_history)])
    
    
    # add dropout
    dpt_model = keras.models.Sequential([
	    keras.layers.Dense(16,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
	    keras.layers.Dropout(0.5),
	    keras.layers.Dense(16,activation=tf.nn.relu),
	    keras.layers.Dropout(0.5),
	    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    
    dpt_model.compile(optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy','binary_crossentropy'])
    
    dpt_model_history = dpt_model.fit(train_data,train_labels,
      epochs=20,
      batch_size=512,
      validation_data=(test_data,test_labels),
      verbose=2)
    
    plot_history([('baseline',baseline_history),
      ('dropout',dpt_model_history)])
    