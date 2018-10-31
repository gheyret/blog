在回归问题中，我们的目标是预测连续值的输出，如价格或概率。 将此与分类问题进行对比，我们的目标是预测离散标签（例如，图片包含苹果或橙色）。

本文构建了一个模型，用于预测20世纪70年代中期波士顿郊区房屋的平均价格。 为此，我们将为模型提供有关郊区的一些数据点，例如犯罪率和当地财产税率。

此示例使用tf.keras API，有关详细信息，请参阅本指南。

    from __future__ import absolute_import, division, print_function
    
    import tensorflow as tf
    from tensorflow import keras
    
    import numpy as np
    
    print(tf.__version__)

## 波士顿住房价格数据集 ##

可以在TensorFlow中直接访问此数据集。 下载并随机播放训练集：

    boston_housing = keras.datasets.boston_housing
    
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    
    # Shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]

### 样本和特征 ###

这个数据集比我们迄今为止使用的其他数据集小得多：它共有506个示例，分为404个训练样例和102个测试示例：

    print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
    print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features    

*Training set: (404, 13)  
Testing set:  (102, 13)*

该数据集包含13个不同的功能：

1. 人均犯罪率。
2. 占地面积超过25,000平方英尺的住宅用地比例。
3. 每个城镇非零售业务的比例。
4. Charles River虚拟变量（如果管道限制河流则= 1;否则为0）。
5. 一氧化氮浓度（每千万份）。
6. 每栋住宅的平均房间数。
7. 1940年以前建造的自住单位比例。
8. 到波士顿五个就业中心的加权距离。
9. 径向高速公路的可达性指数。
10. 每10,000美元的全额物业税率。
11. 城镇的学生与教师比例。
12. 1000 *（Bk - 0.63）** 2其中Bk是城镇黑人的比例。
13. 人口比例较低的百分比。

这些输入数据特征中的每一个都使用不同的比例存储。某些特征由0到1之间的比例表示，其他特征的范围介于1到12之间，有些特征介于0到100之间，依此类推。这通常是现实世界数据的情况，了解如何探索和清理此类数据是一项重要的开发技能。

*关键点：作为建模人员和开发人员，请考虑如何使用此数据以及模型预测可能带来的潜在好处和危害。 像这样的模型可能会加剧社会偏见和差异。 特征是否与您要解决的问题相关或是否会引入偏差？ 有关更多信息，请阅读ML公平性。*

    print(train_data[0])  # Display sample features, notice the different scales

*[7.8750e-02 4.5000e+01 3.4400e+00 0.0000e+00 4.3700e-01 6.7820e+00
 4.1100e+01 3.7886e+00 5.0000e+00 3.9800e+02 1.5200e+01 3.9387e+02
 6.6800e+00]*

使用pandas库在格式良好的表中显示数据集的前几行：

    import pandas as pd
    
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
    'TAX', 'PTRATIO', 'B', 'LSTAT']
    
    df = pd.DataFrame(train_data, columns=column_names)
    df.head()

![](https://i.imgur.com/cYloUz0.png)

### 标签 ###

标签是几千美元的房价。 （您可能会注意到20世纪70年代中期的价格。）

    print(train_labels[0:10])  # Display first 10 entries

*[32.  27.5 32.  23.1 50.  20.6 22.6 36.2 21.8 19.5]*

## 归一化特征 ##

建议对使用不同比例和范围的功能进行归一化。 对于每个特征，减去特征的平均值并除以标准差：

    # Test data is *not* used when calculating the mean and std
    
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    print(train_data[0])  # First training sample, normalized

虽然模型可能在没有特征归一化的情况下收敛，但它使训练更加困难，并且它使得结果模型更依赖于输入中使用的单位的选择。

## 建立模型 ##

让我们建立我们的模型。 在这里，我们将使用具有两个密集连接的隐藏层的Sequential模型，以及返回单个连续值的输出层。 模型构建步骤包含在一个函数build_model中，因为我们稍后将创建第二个模型。

    def build_model():
      	model = keras.Sequential([
    	keras.layers.Dense(64, activation=tf.nn.relu,
      	 input_shape=(train_data.shape[1],)),
    	keras.layers.Dense(64, activation=tf.nn.relu),
    	keras.layers.Dense(1)
     	])
    
      	optimizer = tf.train.RMSPropOptimizer(0.001)
    
      	model.compile(loss='mse',
    			optimizer=optimizer,
    			metrics=['mae'])
      	return model
    
    model = build_model()
    model.summary()

## 训练模型 ##

该模型经过500个周期的训练，并在history对象中记录训练和验证准确性。

    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
    	if epoch % 100 == 0: print('')
    	print('.', end='')
    
    EPOCHS = 500
    
    # Store training stats
    history = model.fit(train_data, train_labels, 						epochs=EPOCHS,
    					validation_split=0.2, verbose=0,
    					callbacks=[PrintDot()])

使用存储在历史对象中的统计数据可视化模型的训练进度。 我们希望使用此数据来确定在模型停止进展之前要训练多长时间。

    import matplotlib.pyplot as plt
        
    def plot_history(history):
      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Mean Abs Error [1000$]')
      plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
       label='Train Loss')
      plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
       label = 'Val loss')
      plt.legend()
      plt.ylim([0, 5])
    
    plot_history(history)

该图表显示在约200个时期之后模型的改进很小。 让我们更新model.fit方法，以便在验证分数没有提高时自动停止训练。 我们将使用一个回调来测试每个时代的训练条件。 如果经过一定数量的时期而没有显示出改进，则自动停止训练。

您可以在此处了解有关此回调的更多信息。

model = build_model()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    
    history = model.fit(train_data, train_labels, 						epochs=EPOCHS,
    					validation_split=0.2, verbose=0,
   						callbacks=[early_stop, PrintDot()])
    
    plot_history(history)

![](https://i.imgur.com/xg54aqc.png)

该图显示平均误差约为2,500美元。 这个好吗？ 好吧，当一些标签只有15,000美元时，2,500美元并不是微不足道的数额。

让我们看看模型在测试集上的表现如何：

    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

*Testing set Mean Abs Error: $2713.16*

## 预测 ##

最后，使用测试集中的数据预测一些房价：

    test_predictions = model.predict(test_data).flatten()
    
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [1000$]')
    plt.ylabel('Predictions [1000$]')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])

![](https://i.imgur.com/JtaFf0U.png)

    error = test_predictions - test_labels
    plt.hist(error, bins = 50)
    plt.xlabel("Prediction Error [1000$]")
    _ = plt.ylabel("Count")

![](https://i.imgur.com/tP157fn.png)

## 结论 ##

本文介绍了一些处理回归问题的技巧。

1. 均方误差（MSE）是用于回归问题的常见损失函数（不同于分类问题）。
2. 同样，用于回归的评估指标也不同于分类。 常见的回归指标是平均绝对误差（MAE）。
3. 当输入数据要素具有不同范围的值时，应单独缩放每个要素。
4. 如果训练数据不多，则选择隐藏层较少的小型网络，以避免过度拟合。
5. 早期停止是防止过度拟合的有用技术。


完整代码：

    from __future__ import absolute_import,division,print_function
    import tensorflow as tf
    from tensorflow import keras
    
    import numpy as np
    
    print(tf.__version__)
    
    # download data
    boston_housing = keras.datasets.boston_housing
    (train_data,train_labels),(test_data,test_labels) = boston_housing.load_data()
    
    # Shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]
    
    print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
    print("Testing set: {}".format(test_data.shape))# 102 examples, 13 features
    print(train_data[0])   # Display sample features, notice the different scales
    
    import pandas as pd
    
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
    'TAX', 'PTRATIO', 'B', 'LSTAT']
    
    df = pd.DataFrame(train_data,columns = column_names)
    print(df.head())
    
    print(train_labels[0:10])   # Display first 10 entries
    
    
    # Normalize features
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) /std
    
    print(train_data[0])
    
    
    # Create the model
    def build_model():
	    model = keras.Sequential([
	    keras.layers.Dense(64,activation=tf.nn.relu,
	       input_shape=(train_data.shape[1],)),
	    keras.layers.Dense(64,activation=tf.nn.relu),
	    keras.layers.Dense(1)
	    ])
	    optimizer = tf.train.RMSPropOptimizer(0.001)
	    
	    model.compile(loss='mse',
	      optimizer=optimizer,
	      metrics=['mae'])
	    return model
    
    model = build_model()
    model.summary()
    
    
    # Train the model
    # Display training progress by printing a single dot for each comploted epoch
    class PrintDot(keras.callbacks.Callback):
	    def on_epoch_end(self,epoch,logs):
		    if epoch % 100 == 0:print('')
		    print('.',end='')
    
    EPOCHS = 500
    # Store training stats
    history = model.fit(train_data,train_labels,epochs = EPOCHS,validation_split=0.2,verbose=0,
    					callbacks=[PrintDot()])
    
    import matplotlib.pyplot as plt
    
    def plot_history(history):
	    plt.figure()
	    plt.xlabel('Epoch')
	    plt.ylabel('Mean Abs Error [1000$]')
	    plt.plot(history.epoch,
	     		np.array(history.history['mean_absolute_error']),
	     		label='Train Loss')
	    plt.plot(history.epoch,
	     		np.array(history.history['val_mean_absolute_error']),
	     		label='Val loss')
	    plt.legend()
	    plt.ylim([0,5])
	    plt.show()
    
    plot_history(history)
    
    
    model = build_model()
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
    history = model.fit(train_data,train_labels,epochs = EPOCHS,
    validation_split=0.2,verbose=0,
    callbacks=[early_stop,PrintDot()])
    plot_history(history)
    
    
    # Predict
    test_predictions = model.predict(test_data).flatten()
    
    plt.scatter(test_labels,test_predictions)
    plt.xlabel('True Values [1000$]')
    plt.ylabel('Predictions [1000$]')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100,100],[-100,100])
    
    plt.show()
    
    
    error = test_predictions - test_labels
    plt.hist(error,bins=50)
    plt.xlabel('Prediction Error [1000$]')
    _ = plt.ylabel('Count')
    
    plt.show()