[原文地址](https://tensorflow.google.cn/tutorials/keras/save_and_restore_models)

可以在训练期间和训练后保存模型进度。 这意味着模型可以从中断的地方恢复，并避免长时间的训练。 保存也意味着您可以共享您的模型，而其他人可以重新创建您的工作。 在发布研究模型和技术时，大多数机器学习从业者分享：

1. 用于创建模型的代码
2. 模型的训练权重或参数

共享此数据有助于其他人了解模型的工作原理，并使用新数据自行尝试。

*注意：小心不受信任的代码 - TensorFlow模型是代码。 有关详细信息，请参阅安全使用TensorFlow。*

## 选项 ##
保存TensorFlow模型有多种方法 - 取决于您使用的API。 本指南使用tf.keras，一个高级API，用于在TensorFlow中构建和训练模型。 有关其他方法，请参阅TensorFlow保存和还原指南或保存在急切中。

## 安装 ##

### 安装和引用 ###

安装和导入TensorFlow和依赖项，有下面两种方式：

1. 命令行：pip install -q h5py pyyaml 
2. 在Anaconda Navigator中安装；

### 下载样本数据集 ###

    from __future__ import absolute_import, division, print_function
    
    import os
    
    import tensorflow as tf
    from tensorflow import keras
    
    tf.__version__

*'1.11.0'*

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
    
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

### 定义模型 ###

让我们构建一个简单的模型，我们将用它来演示保存和加载权重。

    # Returns a short sequential model
    def create_model():
      model = tf.keras.models.Sequential([
	    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
	    keras.layers.Dropout(0.2),
	    keras.layers.Dense(10, activation=tf.nn.softmax)
      ])
      
      model.compile(optimizer=tf.keras.optimizers.Adam(), 
	    loss=tf.keras.losses.sparse_categorical_crossentropy,
	    metrics=['accuracy'])
      
      return model
    
    
    # Create a basic model instance
    model = create_model()
    model.summary()

## 在训练期间保存检查点 ##

主要用例是在训练期间和训练结束时自动保存检查点。 通过这种方式，您可以使用训练有素的模型，而无需重新训练，或者在您离开的地方接受训练 - 以防止训练过程中断。

tf.keras.callbacks.ModelCheckpoint是执行此任务的回调。 回调需要几个参数来配置检查点。

### 检查点回调使用情况 ###

训练模型并将模型传递给ModelCheckpoint：

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
     save_weights_only=True,
     verbose=1)
    
    model = create_model()
    
    model.fit(train_images, train_labels,  epochs = 10, 
      validation_data = (test_images,test_labels),
      callbacks = [cp_callback])  # pass callback to training

这将创建一个TensorFlow检查点文件集合，这些文件在每个时期结束时更新：

    !ls {checkpoint_dir}

*checkpoint  cp.ckpt.data-00000-of-00001  cp.ckpt.index*

创建一个新的未经训练的模型。 仅从权重还原模型时，必须具有与原始模型具有相同体系结构的模型。 由于它是相同的模型架构，我们可以共享权重，尽管它是模型的不同实例。

现在重建一个新的未经训练的模型，并在测试集上进行评估。 未经训练的模型将在偶然水平上执行（准确度约为10％）：

    model = create_model()
    
    loss, acc = model.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

然后从检查点加载权重，并重新评估：

    model.load_weights(checkpoint_path)
    loss,acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

*1000/1000 [==============================] - 0s 40us/step
Restored model, accuracy: 87.60%*

### 检查点回调选项 ###

回调提供了几个选项，可以为生成的检查点提供唯一的名称，并调整检查点频率。

训练一个新模型，每5个时期保存一次唯一命名的检查点：

    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
	    checkpoint_path, verbose=1, save_weights_only=True,
	    # Save weights, every 5-epochs.
	    period=5)
    
    model = create_model()
    model.fit(train_images, train_labels,
      epochs = 50, callbacks = [cp_callback],
      validation_data = (test_images,test_labels),
      verbose=0)

现在，查看生成的检查点并选择最新的检查点：

    ! ls {checkpoint_dir}

*checkpoint            cp-0030.ckpt.data-00000-of-00001
cp-0005.ckpt.data-00000-of-00001  cp-0030.ckpt.index
cp-0005.ckpt.index        cp-0035.ckpt.data-00000-of-00001
cp-0010.ckpt.data-00000-of-00001  cp-0035.ckpt.index
cp-0010.ckpt.index        cp-0040.ckpt.data-00000-of-00001
cp-0015.ckpt.data-00000-of-00001  cp-0040.ckpt.index
cp-0015.ckpt.index        cp-0045.ckpt.data-00000-of-00001
cp-0020.ckpt.data-00000-of-00001  cp-0045.ckpt.index
cp-0020.ckpt.index        cp-0050.ckpt.data-00000-of-00001
cp-0025.ckpt.data-00000-of-00001  cp-0050.ckpt.index
cp-0025.ckpt.index*

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    latest

*'training_2/cp-0050.ckpt'*

注意：默认的tensorflow格式仅保存最近的5个检查点。

要测试，请重置模型并加载最新的检查点：

    model = create_model()
    model.load_weights(latest)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

*1000/1000 [==============================] - 0s 96us/step
Restored model, accuracy: 86.80%*

## 这些文件是什么？ ##

上述代码将权重存储到检查点格式的文件集合中，这些文件仅包含二进制格式的训练权重。 检查点包含：*一个或多个包含模型权重的分片。 *索引文件，指示哪些权重存储在哪个分片中。

如果您只在一台机器上训练模型，那么您将有一个带有后缀的分片：.data-00000-of-00001

    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')
    
    # Restore the weights
    model = create_model()
    model.load_weights('./checkpoints/my_checkpoint')
    
    loss,acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

## 保存整个模型 ##

整个模型可以保存到包含权重值，模型配置甚至优化器配置的文件中。 这允许您检查模型并稍后从完全相同的状态恢复培训 - 无需访问原始代码。

在Keras中保存功能齐全的模型非常有用 - 您可以在TensorFlow.js中加载它们，然后在Web浏览器中训练和运行它们。

Keras使用HDF5标准提供基本保存格式。 出于我们的目的，可以将保存的模型视为单个二进制blob。

    model = create_model()
    
    model.fit(train_images, train_labels, epochs=5)
    
    # Save entire model to a HDF5 file
    model.save('my_model.h5')

*Epoch 1/5
1000/1000 [==============================] - 0s 395us/step - loss: 1.1260 - acc: 0.6870
Epoch 2/5
1000/1000 [==============================] - 0s 135us/step - loss: 0.4136 - acc: 0.8760
Epoch 3/5
1000/1000 [==============================] - 0s 138us/step - loss: 0.2811 - acc: 0.9280
Epoch 4/5
1000/1000 [==============================] - 0s 153us/step - loss: 0.2078 - acc: 0.9480
Epoch 5/5
1000/1000 [==============================] - 0s 154us/step - loss: 0.1452 - acc: 0.9750*

现在从该文件重新创建模型：

    # Recreate the exact same model, including weights and optimizer.
    new_model = keras.models.load_model('my_model.h5')
    new_model.summary()

检查其准确性：

    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

这项技术可以保存以下：

1. 权重值
2. 模型的配置（架构）
3. 优化器配置

Keras通过检查架构来保存模型。 目前，它无法保存TensorFlow优化器（来自tf.train）。 使用这些时，您需要在加载后重新编译模型，并且您将失去优化器的状态。

## 下一步是什么 ##

这是使用tf.keras保存和加载的快速指南。

tf.keras指南显示了有关使用tf.keras保存和加载模型的更多信息。

请参阅在急切执行期间保存以备保存。

“保存和还原”指南包含有关TensorFlow保存的低级详细信息。


完整代码：

    from __future__ import absolute_import,division,print_function
    import os
    import tensorflow as tf
    from tensorflow import keras
    
    print(tf.__version__)
    
    
    # Download dataset
    (train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
    
    train_images = train_images[:1000].reshape(-1,28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1,28 * 28) / 255.0
    
    # Define a model
    # Returns a short sequential model
    def create_model():
	    model = tf.keras.models.Sequential([
	    keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(784,)),
	    keras.layers.Dropout(0.2),
	    keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      metrics=['accuracy'])
    return model
    
    # Create a basic model instance
    model = create_model()
    model.summary()
    
    # Checkpoint callback usage
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
     save_weights_only=True,
     verbose=1)
    model = create_model()
    model.fit(train_images,train_labels,epochs=10,
      validation_data=(test_images,test_labels),
      callbacks=[cp_callback]) # pass callback to training
    
    # Create a new, untrained model. 
    model = create_model()
    loss,acc = model.evaluate(test_images,test_labels)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
    
    # Load the weights from chekpoint, and re-evaluate.
    model.load_weights(checkpoint_path)
    loss,acc = model.evaluate(test_images,test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    
    # Train a new model, and save uniquely named checkpoints once every 5epochs
    # include the epoch in the file name. (uses 'str.format')
    checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
	    checkpoint_path, verbose=1,save_weights_only=True,
	    # Save weights, every 5-epochs
	    period=5)
    
    model = create_model()
    model.fit(train_images,train_labels,
      epochs=50,callbacks = [cp_callback],
      validation_data = (test_images,test_labels),
      verbose=0)
    
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    
    
    # To test, reset the model and load the latest checkpoint
    model = create_model()
    model.load_weights(latest)
    loss, acc = model.evaluate(test_images,test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    
    # Manually save weights
    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')
    # Restore the weights
    model = create_model()
    model.load_weights('./checkpoints/my_checkpoint')
    
    loss, acc = model.evaluate(test_images,test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    
    
    # Save the entire model
    model = create_model()
    model.fit(train_images,train_labels,
      epochs=5)
    # Save entire model to a HDF5 file
    model.save('my_model.h5')
    
    # Recreate the exact same model, including weights and optimizer.
    new_model = keras.models.load_model('my_model.h5')
    new_model.summary()
    
    loss, acc = new_model.evaluate(test_images,test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))