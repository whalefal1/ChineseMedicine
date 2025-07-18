# 中医舌诊

## 项目背景

舌诊是中医诊断疾病的重要手段之一。目前，临床舌诊主要依靠医生观察病人舌头并通过经验进行分析诊断，对医生的专业以及经验都有较高的要求。此外，仅仅通过医生肉眼观察，并根据经验做出诊断的过程无法实现数字化的记录，不利于中医舌诊的发展

## 项目需求

本项目旨在实现一个自动化的中医舌诊系统，用户只需输入一张舌体图像，就能获得心、肝、脾、肺、肾等脏器的健康指数以及总体健康指数，进而对用户的健康状况做出判定，实现效果如下图所示:
![image-20250624162011995](https://whalefal1.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E5%BA%8A/202506241620071.png)

### 目标描述

输入：通过摄像头拍摄的彩色舌体图像输出：主要包含以下两部分的内容：

·心、肝、脾、肺、肾等脏器的健康值以及总体健康值；

·用户的健康状况，包含健康、血虚、脾虚、肾虚、气虚、肝郁6种情况的判定。

舌诊的业务逻辑：系统可自动从舌体图像中定位出舌体，结合图像特征以及专家经验计算心、肝、脾、肺、肾等对应的健康值及总体健康值，并将最终的判定结果返回四、数据说明

4.1舌体图像

·数据描述：该数据集为真实摄像头拍摄的舌体图像数据

·数据量描述：

。训练数据量：931张。测试数据量：115张。样例数据量：8张

·数据位置描述：

。训练和测试数据位置：https://staticfile.eduplus.net/dataSet/systemLib/4a3ed1f9d210445a9b209c443bc154f5.zip 。样例数据位置：【资料】中的example.zip压缩包

·数据内容描述：本项目提供的是舌体数据集，保存在tongue_data/目录下，目录结构如下图所示，包含train_img、train label、test_img、test_label四个子目录。 

train_img目录：存放931张舌体图像，作为训练数据。部分舌体图像如下图所示：

train_label目录：存放931张舌体图像对应的标签。上述四幅舌体图像对应的标签图像如下图所示。标签图像中属于舌体部分的像素值为1，非舌体部分的像素值为test_img目录：存放115张舌体图像，作为测试数据。

test_label目录：存放115张舌体图像对应的标签图像，像素值分布与训练数据标签图像相同。测试样例描述：

每个测试样例包含舌体图像和专家判定结果两部分，如下图所示：

![image-20250624162228389](https://whalefal1.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E5%BA%8A/202506241622438.png)

4.2专家经验权重

除上述舌体图像外，本项目还提供了心、肝、脾、肺、肾等脏器对应的特征权重。数据描述：心、肝、脾、肺、肾等脏器对应的特征权重

数据位置：【资料】中的LS.csv文件

数据内容：特征权重是根据931幅训练图像的特征计算得到的，具体计算过程如下：

·将图像转换成Lab、HLS、RGB三种格式；

·分别计算三种格式图像舌体各脏器对应区域各通道的直方图，每种格式的图像都有3个通道，即每个脏器区域共需要计算9个通道的直方图；

·根据直方图计算最大值（峰值）、标准差宽度、均值、截断均值；

·至此，每幅每个脏器区域共得到36个特征值，5个脏器区域共180个特征；对于931幅图像，得到931×180的特征矩阵。

·最后，用LaplacianScore方法识别并选择出最能表达数据集特征的特征集，得到1x180维特征。

LS.csv文件共包含360列数据，一次计算产生180列数据，提供了2次计算的结果；注：实际使用中只取后180列数据即可。每36个数据表示一个脏器的对应的特征权重，如下表所示：

![image-20250624162236792](https://whalefal1.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E5%BA%8A/202506241622824.png)

数据示例：

部分数据示例如下表所示：

![image-20250624162319446](https://whalefal1.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E5%BA%8A/202506241623480.png)

### 系统测试

本项目提供8个测试样例，6个一样测试样例的判定结果与专家判定结果一致，则满足使用需求

## 数据加载与处理

![image-20250624172550610](https://whalefal1.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E5%BA%8A/202506241725661.png)

本任务主要定义数据加载及处理的函数。在"data_loader.py"文件中编写本任务的代码。

(1)导入包

(2)设置随机种子

设置随机种子，使得随机数据可预测，即只要seed的值一样，后续生成的随机数都一样。

（3）获取图片的路径

定义DataLoadError类，当预训练模型下载出现异常，致使接下来的训练不能进行时，DataLoadError类会抛出异常。

定义get_pairs_from_paths（）方法，在检查数据完整性的同时，查找目录下的所有舌体图像文件，以及对应的标签文件。该方法的主要步骤为：

·遍历所有的图像，如果图像的格式符合要求，则将将图像的文件名、扩展名、图像的路径作为一个对象保存在image_files列表中；

·遍历所有的标签图像，如果图像的格式符合要求，则将该标签文件的文件扩展名及文件路径保存在标签文件中；

·匹配图像及其标签文件，如果匹配成功，则将图像的路径信息和标签图像的路径信息添加到列表中；如果匹配不成功，则抛出异常。

（4）标准化处理图像

定义get_image_array函数，该方法用于读取并标准化处理图像。该方法的主要实现流程为：

·如果传入的image_input参数是str类型，则使用cv2.imread()方法读取图像；

·根据传入的imgNorm参数，选择标准化处理图像的方式。

（5）标准化处理标签图像

定义get_segmentation_array函数，标准化处理标签图像。该方法的主要实现流程为：

·如果传入的参数image_input是str类型，则使用cv2.imread()方法读取图像；

·将标签图像调整到标准尺寸，使用最邻近插值的方法；

·获取标签图像的第0通道，如果像素值大于1，则赋值为1

·根据no_reshape参数，确定是否需要改变数据的维度。

（6）创建数据迭代器

定义image_segmentation_generator函数，其作用是创建一个数据迭代器。image_segmentation_generator()的主要实现流程如下：

·调用定义的get_pairs_from_paths（）方法，在检查数据完整性的同时，查找目录下的所有舌体图像文件，以及对应的标签文件；

·随机打乱数据的顺序；

·使用itertools.cycle()方法，创建一个迭代对象，对于输入的iterable元素反复执行循环操作。eg:itertools.cycle("ABCD")>ABCDABCD..;

·遍历每个batch_size中的图像；

·将读取的图像数据进行标准化处理后，存入X列表中；

·将读取的标签数据进行标准化处理后，存入Y列表中。





## 图像语义分割模型设计

### 1.开发准备

(1)创建子目录在ChineseMedicine/keras_.segmentation"目录下创建子目录"models"。
(2)创建代码文件在ChineseMedicine/.keras_segmentation/models目录下新建五个python文件，分别命名为"_init_py、"resnet50.py、"unet.py、"model_utils..py、"all_models.py。

### 2.构建Res-Unet网络模型本任务主要完成Res-Unet网络模型的构建。

#### 2.1构建编码器本模块主要完成编码器的构建，在keras_.segmentation/models/目录下的"resnet50.py"文件中添加本模块的代码.

(1)导入库

(2)设置相关参数·IMAGE_ORDERING:该参数用来设定数据的维度顺序。在深度学习中，不同的深度学习框架可能对应不同的影像表达，在数据处理时应做相应的转换。在表示一组彩色图片的问题上，Theano和Cafe使用（样本数，通道数，行或称为高，列或称为宽）通道在前的方式，称为channels_.frst;而TensorFlow使用（样本数，行或称为高，列或称为宽，通道数)通道在后的方式，称为channels_last。pretrained_url:resnets5O的预训练模型的下载路径。
(3)one_side填充定义one side_pad0方法，该方法主要是为了实现在特征图的上侧和左侧各补充一行（一列）0，特征图的行数+1，列数+1。
该方法的主要步骤如下：·首先，使用ZeroPadding.2D0方法为特征图的上下左右各补充一行一列0，特征图的行数+2，列数+2；·然后，将右侧和下侧补充的0去除，就实现了one_side填充。
(4)定义Identity Block残差块在本模块中，定义identity_.block函数，即定义一个输入和输出维度相同的残差块，该残差块的作用是较为稳定地通过加深层数来提高模型的效果，同时也可以避免梯度消失的问题。
(5)Conv Block残差块定义co_block函数，实现一个残差块，输入和输出的维度是不一样的，所以不能连续串联，它的作用是改变网络的维度(6)实现编码器搭建好组件残差块之后就是确定网络结构，将一个个残差块组成残差网络。

#### 2.2定义模型的属性和方法本模块定义模型的一些属性和方法keras_segmentation/models/目录下的"model_utiIs.py文件中添加本模块的代码。

(1)导入包(2)定义模型的属性和方法定义get_segmentation_model函数，定义模型的一些属性和方法。

#### 2.3实现res-unet网络本模块实现res-unet网络的搭建，在keras_.segmentation/models/目录下的"unet.py"文件中添加本模块的代码。

(1)导入包(2)设置相关参数·IMAGE_ORDERING:该参数用来设定数据的维度顺序。在深度学习中，不同的深度学习框架可能对应不同的影像表达，在数据处理时应做相应的转换。在表示一组彩色图片的问题上，Theano和Caffe使用（样本数，通道数，行或称为高，列或称为宽）通道在前的方式，称为channels_first;而TensorFlow使用（样本数，行或称为高，列或称为宽，通道数)通道在后的方式，称为channels_last。
·MERGE_AXIS:值为-1，表示在最后一维操作(3)构建解码器定义unet函数，构建解码器。
(4)实现Res-Unet模型定义resnets50_unet函数，组合编码器和解码器，以实现Res-Unet模型。
2.4其他在keras_segmentation/models/.目录下的al_models.py文件中添加传入模型名称的代码。
将nodels/目录封装成一个Pythont模块。_init_py文件的作用是将当前文件夹变为一个Python模块，在导入一个包时，实际上是导入了它的_init_py文件。这样可以在_it_py文件中批量导入所需要的模块，而不再需要一个一个的导入。

## 训练语义分割模型

### 1.开发准备

(1)创建代码文件
在ChineseMedicine/根目录下新建一个python文件，命名为""main_train.py。
在ChineseMedicine/keras_.segmentation/目录下新建两个python文件，分别命名为"train.py、"_init_.py。
(2)新建子目录
在ChineseMedicine/根目录下新建一个子目录，命名为weights",用于存储训练好的模型。

### 2.训练Res-Unet网络模型

本任务主要完成Res-Unet网络模型的训练。

#### 2.1定义训练过程函数

本模块主要定义训川练过程函数，以便后续完成模型的训练。本部分的代码在keras segmentation/目录下的train.py文件中进行编写。
(1)导入包
(2)查找训练轮次最多的模型
定义find_latest_checkpointi函数，该函数的作用是查找训练轮次最多的模型，一般而言，被训练轮次越多的模型预测效果越好。
find_latest_checkpoint(0函数的主要实现流程如下：
·首先，定义get_epoch._number_.from_path(0函数，用于分割模型的路径，获得路径的单词列表
·然后，判断是否能匹配到相应的文件，如果能找到模型文件，则根据模型中的数字编号，找到训练轮次最多的模型。
(3)计算损失值
定义masked_categorical_crossentropy()函数，使用categorical_crossentropy0方法计算损失值。
(4)保存训练过程中的模型
定义CheckpointsCallback类，将训练过程中的模型都保存在指定目录下。
(⑤)定义训练过程的方法
定义train()方法，在该方法中，定义训练过程中需要的参数，以及模型的训练过程。主要流程如下：
·定义模型训练过程中需要的多个参数，如epoch、bach_size、优化器等；
·编译模型，确定训练模型使用的损失函数、优化器、评价指标等：
·判断是否存在训练过的模型，如果存在，则在训练过的模型的基础上继续进行训练；
·对训练集和验证集数据进行数据增强，使用ft_generator(0方法训练模型，并保存每个epoch训练完成的模型。
完成上述代码的编写后，需要将keras_.segmentation目录封装成一个Pythont模块。

#### 2.2训练模型

该部分主要实现模型训练，在ChineseMedicine/目录下的"main train.py"文件中编写本模块代码。
(1)导入包
(2)实现模型的训练
在该部分中，主要调用定义好的resnet50unet网络结构，以及调用train(O函数，并传入各参数，实现模型的训练。
代码编写完成后，可在终端或者Pycharm中运行nain_train.py文件。
代码执行过程中，首先会下载预训练的且无头部（无全连接层）的resnet50模型，如下图所示：
预训练模型下载完成后，会在其基础上继续进行训练，训练过程如下图所示，训练30个epochs.大概需要1小时。

## 评估图像语义分割模型

### 1.开发准备

#### (1)创建代码文件

在ChineseMedicine/根目录下新建两个python文件，分别命名为"test.py、"comparison.py。 在ChineseMedicine/keras_segmentation/目录下新建一个python文件，命名为"predict.py。 (2)新建子目录

在ChineseMedicine/根目录下新建两个子目录，分别命名为"prediction"、"compare"。 2.测试模型

本任务主要是对训练好的模型进行测试，以评估模型的性能2.1定义测试函数

在keras segmentation/目录下的"predict.py"文件中编写本模块代码。 (1)导入包

### (2)加载模型

定义model_from_checkpoint_path()方法，该方法主要用于加载训练轮次最多的模型。model_from_checkpoint_path()方法的具体实现流程：

先判断resunet_config.json"是否为文件；其次加载resunet_config.json文件的内容，然后查找出被训练轮次最多的模型，将训练好的模型权重载入模型。（3）获取标签图像颜色数据

定义get_colored_segmentation_image（）函数，该函数的作用是获取输出标签图像的颜色数据，将标签图像转换成一个288*388*3的数组。

### （4）可视化分割结果

定义visualize_segmentation（）函数，突出标签中舌体的位置，便于人眼直接查看。

### (5)模型预测

定义predict（）函数，该方法的主要作用是调用训练好的模型，对图像进行预测。主要实现流程：

·使用cv2.imread（）方法读取图像；

·调用定义的get_image_array（）方法，对输入图像进行标准化处理；

·调用训练好的模型对图像进行预测；

·调用定义的visualize_segmentation()方法，可视化舌体分割图像；

·保存分割图像。

### (6)定义评价指标

定义evaluate（）函数，用于计算评价指标miou，评估模型的分割性能。2.2实现模型的测试

在该部分主要实现模型的测试，在主目录下的test.py文件中编写本模块代码。

#### (1)导入包

#### (2)加载模型

定义get_model（）函数，加载训练好的模型。(3)模型测试函数

定义test_img（）函数，用训练好的模型对输入的图像进行测试；test_img的具体实现流程：

使用listdir（）方法获取测试图像目录下的所有文件。调用evaluate（）函数对模型的测试结果进行评价，输出MIOU结果。（4）定义测试主程序

定义main（）函数，调用上述定义的函数，实现模型的测试。代码编写完成后，可在终端或Pycharm中运行test.py文件。

代码执行结束后，会计算loU指标，以评估模型的性能

![image-20250625093100417](https://whalefal1.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E5%BA%8A/202506250931477.png)

IoU的值在96%以上，说明模型的性能满足使用的需求。注：同学们的结果可能与上述结果不相同，这主要是因为数据标注、数据集划分以及测试集的不同导致的。可视化的舌体分割结果，保存在prediction/目录下，如下图所示：

![](https://whalefal1.oss-cn-beijing.aliyuncs.com/图床/202506250931574.png)

### 3.测试结果对比

在本部分，主要将原图与预测结果图放到一张图上进行对比，查看预测效果。

在"comparison.py"文件中编写本模块代码。

#### (1)导入包

#### (2)设置全局变量

设置全局变量，用于指定测试图像的路径、预测图像的路径以及合并后的图像保存的路径。

#### (3)结果对比

在本部分，将测试图像及其预测结果图像合并在一起，并将合并厚土图像保存到compare目录下，以便于进行对比。主要过程如下：

·遍历测试图像；使用plt.subplot（）方法创建一个1行两列的图像，第一列放置测试图像，第2列放置预测图像；将合并后的图像保存到compare目录下。

代码编写完成后，可在终端或Pycharm中运行comparison.py文件。

对比结果保存在compare/目录下，效果如下图所示：

![image-20250625093207988](https://whalefal1.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E5%BA%8A/202506250932052.png)

从上图的比对中，可以发现不同的光线及图像的清晰度会影响对舌体的分割，模型对图像中舌体的位置定位的很准确，图像中舌体的形状各有不同，但都能准确分割出舌

体。

## 舌体分析

### 舌体区域分割

本任务主要定义该部分主要定义将舌面划分为肺，脾，肾，左肝，右肝五部分脏器对应的区域的函数。知识点：最小正外接矩形、二阶贝塞尔曲线

重点：最小正外接矩形、二阶贝塞尔曲线难点：最小正外接矩形、二阶贝塞尔曲线

### 1.开发准备 

### (1)创建子目录

在"ChineseMedicine/"目录下创建子目录"tongue"。

在ChineseMedicine/tongue"目录下创建子目录"tongue_segmentation"。 (2)创建代码文件

在ChineseMedicine/tongue/tongue_segmentation目录下新建一个python文件，命名为"segmentation.py"。 在ChineseMedicine/tongue/目录下新建一个python文件，命名为"segmentation_tongue.py"。

### 2.定义将舌面划分为各脏器对应的区域的函数

该部分主要定义将舌面划分为肺，脾，肾，左肝，右肝五部分脏器对应的区域的函数，在ChineseMedicine/tongue/tongue_segmentation目录下的"segmentation.py"文件中编 写本模块的代码。

#### (1)导入包

#### （2）获取舌头的最小正外接矩形

定义get_tongue（）函数，获取舌头的最小正外接矩形。

#### （3）利用二阶贝塞尔曲线

定义Bezier2（）函数，利用二阶贝塞尔曲线将不规则的舌体从正外接矩阵中进一步定位。（4）将舌面划分为各脏器对应的区域

定义vscera_split（）函数，该方法的作用是将分割出的舌体划分为心肺，脾，肾，左肝，右肝五部分，在图像中确定各个区域的部分，如下图所示：

舌根肾

舌中)脾胃

舌尖心肺 

### 3.实现舌面分割

该部分主要调用segmentation.py文件中定义的函数，实现舌面分割，在tongue/目录下的segmentation_tongue.py文件中编写本模块代码。

####  (1)导入包

#### (2)设置相关参数

·graph:使用tf.get_default_graph()获取当前默认计算图；

·model：找到最终训练好的模型。

#### (3)舌面分割

定义seg_tongue（）函数，以实现将舌面划分为各脏器对应的区域。该函数的具体流程如下：

·首先，调用训练好的模型，对舌体图像进行分割，得到分割结果图像；

·然后，找到分割图像中的舌体区域；

·最后，调用定义的viscera_split（）函数，将舌体划分为各脏器对应的区域。

## 舌部质量检测

### 任务描述

本任务主要定义舌部质量检测的函数，若出现图像过亮、过暗、未检测到舌体以及检测到的舌体不完整等情况，则认为图像质量检测不通过，需重新拍摄图像。
知识点：计算图像的亮度、计算舌体占图像的比例
重点：计算图像的亮度、计算舌体占图像的比例
难点：计算图像的亮度、计算舌体占图像的比例
内容：1.开发准备：创建代码文件

### 2.舌部质量检测

#### (1)导入包

#### (2)设置全局变量

#### (3)计算图像的亮度以及舌部所占整张图的比例

#### (4)舌部质量检测

### 任务指导

#### 1.开发准备

##### (1)创建代码文件

在ChineseMedicine/.tongue//目录下新建一个python.文件，命名为haveTongue.py"。

#### 2.舌部质量检测

本任务的作用是对图像进行舌部质量检测，若出现图像过亮、过暗、未检测到恬体以及检测到的舌体不完整等情况，则认为图像质量检测不通过，需重新拍摄图像。
在/tongue/.目录下的naveTongue.py文件中编写本模块代码。

##### (1)导入包

##### (2)设置全局变量

设置面积控制阈值，以便于后续用于判断图像质量检测是否通过。

##### (3)计算图像的亮度以及舌部所占整张图的比例

定义calcuAera(O函数，在该方法中计算图像的亮度以及舌部所占整张图的比例，以用于判断图像的质量。calcuAera0函数的具体实现流程如下：
·读取图像，并使用cv2.cvtColor(0方法将图像转换为灰度图像：
·求图像中所有像素的和，并计算像素的平均值，如果像素值平均值小于50，则表示亮度过暗，如果像素的平均值大于200，则表示图像过亮；
·调用定义的seg_tongue(0函数，获取舌体mask,根据mask图像获取舌体的最小矩形框，从灰度图像中提取矩形框中的区域，并计算矩形框区域中像素的平均值，进一步判
定图像是否过亮或者过暗：
·根据矩形框进一步计算舌部所占整张图的比例。

##### (4)舌部质量检测

定义haveTongue(0函数，最终对图像质量进行判断。图像质量检测不通过，主要包含以下三种情况：
·图像过暗：
·图像过亮；
·舌体占整张图像的比例小于设置的最小面积阈值，或者舌体占整张图像的比例大于设置的最大面积阈值，则表示未检测到舌体或者舌体不完整。

## 计算舌部各区域的特征值及占比

### 任务描述

本任务主要计算舌部各区域的特征值以及占比。
知识点：最大值（峰值）、标准差宽度、均值、截断均值、色彩空间
重点：计算特征值
难点：计算特征值

### 内容

1.开发准备：创健代码文件
2.计算舌部各区域的特征值以及占比
(1)导入包
(2)求最大值（峰值）、标准差宽度、均值、截断均值
(3)遍历色彩空间计算特征值
(4)计算舌部各区域的特征值以及占比
任务指导
1.开发准备
(1)创建代码文件
在ChineseMedicine/tongue/.目录下新建一个python文件，命名为"tongueHist.py"。
2.计算舌部各区域的特征值以及占比
在本任务中，主要计算舌部特征值与各区域的比例，在tongue/目录下的tongueHist.py文件中编写本模块代码。
(1)导入包
(2)求最大值（峰值）、标准差宽度、均值、截断均值
定义getStatistics(0函数，该函数根据各通道的直方图计算最大值（峰值）、标准差宽度、均值、截断均值。
(3)遍历色彩空间计算特征值
定义calcuVec()函数，遍历色彩空间计算特征值。calcuVec函数的具体实现流程如下：
·遍历颜色空间，将BGR格式的图像分别转换为Lab、HLS、RGB格式；
·遍历每种格试的通道，使用cv2.calcHist0方法计算各通道的直方图；
·调用定义的getStatistics0函数，根据直方图，计算最大值（峰值）、标准差宽度、均值、截断均值。
(4)计算香部各区域的特征值以及占比
定义getVec(0函数，计算舌部各区域的特征值以及占比。getVec(0方法的主要实现流程如下：
·调用定义的seg_tongue0方法获取心肺、肝（左）、肝（右）、肾、脾对应的区域以及舌体mask;
·统计各区域的像素点数量以及占总量的比值：
。
调用定义的calcuVec(O函数，计算舌部各区域的特征值。

## 计算舌体各区域健康值


本任务主要计算舌部各区域对应的健康得分。
知识点：preprocessing、numpy
重点：preprocessing、.numpy
难点：preprocessing、numpy
内容：1.开发准备：创健子目录、代码文件
2.计算各区域的健康值
(1)定义特征融合的函数
(2)特征融合主函数
(3)实现各区域健康值的计算
任务指导
1.开发准备
(1)创建子目录
在ChineseMedicine/"目录下创建子目录"analysis"。
(2)创建代码文件
在ChineseMedicine/.analysis/目录下新建辆个python文件，分别命名为merge_features.py、"main.py。
在ChineseMedicine目录下新建一个python.文件，命名为ChineseMedicine_.analysis.py"。
(3)上传特征文件
从资料中下载LS.csv文件，并上传至"analysis/"目录下。
2.计算各区域的健康值
2.1定义特征融合的函数
本模块主要用来计算图像中舌体各区域的健康值，在/analysis//目录下的merge_features.py文件中添加本模块的代码。
(1)导入包
(2)特征融合
定义merge_feature(0函数，根据经验权重，融合特征。特征融合过程如下：
首先，使用np.multiply(0方法，计算features和score两个数组对应位置元素相乘的结果；然后，使用np.sumO方法计算对应位置元素相乘的结果的和。
(3)归一化区域得分
定义scaler_feature(0函数，该函数的作用是归一化处理各区域的得分，将所有的特征值缩放到0~1之间。
(4)融合区域特征
定义merge_.region()函数，该方法的作用是根据区域的占比，融合所有区域的得分。
(⑤)健康得分
定义health_score(O函数，该函数的作用是将每一次健康得分，映射到【-1,1]之间。
2.2特征融合主函数
本模块主要通过调用analysis/.merge_features..py文件中的函数来实现健康得分的计算，在/analysis/目录下的main.py文件中编写本模块代码。
(1)导入包
(2)获取当前目录的路径
(3)计算各区域健康得分以及整体得分
定义Main()函数，计算舌部对应的心、脾、肾、肺、肝五个区域的健康值得分及整体健康值得分。
2.3.实现各区域健康值的计算
在ChineseMedicine/目录下的ChineseMedicine_analysis.py文件中编写本模块代码。
(1)导入包
(2)设置相关参数
(3)计算各区域的健康值
定义analysis_ChineseMedicine(O函数，用于计算各区域的健康值，并写入vaIue.tt文件中。流程如下：
·调用定义的getVec()方法，获取舌体的特征值、各区域所占比值、各区域的特征值：
·通过用户的id判断features.文件夹中是否有该用户的对应的子目录，若没有，则创建一个，然后判断该用户是否有历史健康值，如果没有，将计算的各区域健康值写入
value..bxt文件中；如果存在，先读取历史健康值，然后再计算本次检测的各区域健康值，并将计算结果写入value.txt文件中。

## 分析用户健康情况

任务10：舌体分析一分析用户的健康状况
童议工时：5
任务描述
本任务是调用上述任务中定义的函数，对检测到的舌部区域进行分析，得出整体使章值得分以及心、肝、牌、肺、肾名区域的健童值得分，得分区间是卧1,1】，越靠近0越
好。
知识点：函数调用、Opencv
三点：函数调用、Opencv
难点：函数调用、Opencv
内容：1开发准备：代码文件
2.分析用户的健章状况
(1)导入包
(2)设量相关参数
(3)定义型除数据的函数
(4)调整并保存数据，返回缩放比字
(5)舌体图像质量检测
(6)计算健廣得分
(7)主必数
任务指导
1.开发准备
(1)创建代码文件
在ChineseMedicine/根目录下新建一个lupyter文件，命名为run.ipynb,
(2)下载阅试数据
从资料中下载example.zip压缩包，上传并解压至ChineseMedicine/根目录下。
2.分折用户的健康状况
本模块作用是输出用户的健亲分值，在ChineseMedicine/根目录下的run.ipynb.文件中编写本棋块代码。
(1)导入包
(2)设置相关参数
(3)定义删除数据的函数
定义rm_img(0函数，用于型除旧文件。
该涵数的具体实现流程如下：
·调用exists(0方法类断文件是香存在，如果存在，再通过remove0方法移除】
(4)调整并保存数据，返回缩放比率
定义save_mg0函做，用于调密图像尺寸，并保存图像，具体实现流程如下：
·语用定义的m_img0函数，移除旧文件：
读取图像，获取图像的宽和高，并计算缩放比例：
·如果缩放比大于1，则使用cv2.esize(方法调整图像尺寸，并保存图像，返回缩放比例、图像的完和高
·反之，如果缩放比小于等于1，则不调整图像的尺寸，直接保存图像，
(5)舌体因像质量检测
定义ind_tongue(0函数，对舌部图像进行质量检测。find_tongue函数的具体实现流程徵如下：
·调用定义的save_img0函数，调整图像的尺寸：
·调用在haveTongue.py文件中定义的haveTongue()函数，对括部图像进行质量检测，如果图像质量不合格，则返回异常信总，并移除文件；如果图像质量合格，则
输出"code、"box、"msg和time_consuming信点，以及舌部mask,
(6)计算链康得分
定义analysis(0硒数，该函数的主要作用是对检测到的舌部区域进行分析，得出整体链章值得分以及心、肝、脾肺、图各区域的健章值得分，得分区间-1，】，越靠
近0越好。analysis的具体实现流程：
·首先，调用定义的s3 ve_img0方法，调整图像尺寸，并保存图像：
·然后，用ChineseMedicine_.analysis..py文件中的analysis_ChineseMedicine(0函数，计算整体以及客区域的链章值，并写入vaIue..tt文件文件。
(7)主函数
定义主函数，以总装括体分析整个过程中的函数，实现用户的使章状况分析。主要实现流程如下：
·输入图像，并调用定义的find_tongue(函数，对舌部图像进行质量检测，图像质重合格，会输出舌体矩形的左上角和右下角的坐标，以及舌体mask;
根据坐标信总，从原图像中获取舌体区域，并将舌部mask图像从单通道转为三通道，然后将原图与预测的m35k图拼接在一起；
·调用定义的analysis0函数，对舌部进行分析，得出整体健亲值得分以及对应心肝脾肺肾名部分的链奈值得分：
·并依据专家经验以及多次实验分析设定镜位，将计算得到的健亲值得分和雨值作较，以判断用户的健章状况，包含链章、血应、脾应、肾虚、气虚、肝郁6种情况
的判定，
代码编写完成后，在奖端或Pycharmi运行run.py文件。
程字执行过程中，会显示原图与舌部ask图像拼接在一起的图像，如下图所示：

在图像上，按任意键关闭图像后，终端页面会继续输出如下图所示：

输出结果是一个字典，字典中的healthy值表示整体健康值，heart值表示心健康值，spleen值表示脾健康值，kidney值表示肾健康值，lung值表示肺健康值，liver值表示
肝健康值。根据设定的阈值判断，该用户的身体存在"血虚"和"气虚"的问题。
但由于该用户是第一次进行舌诊，系统中并没有该用户的历史数据，在处理时，将该用户各脏器的健康值均预设为了0.0，即假定该用户是健康的，因此判断结果会略有
偏差，为了获取更准确地结果，可以多次重复执行，一般第二次舌诊的结果就很准确了。
重新运行un.py文件，结果如下图所示：
第二次诊断该用户的身体存在"血虚"、"脾虚"和"气虚"问题。
打开example/目录下的expert_diagnosis.Xlsx文件，查看0.jpg对应的专家诊断结果如下图所示：
专家对于该用户的诊断结果和舌诊系统第二次判断的结果一致。
此外，用户使用舌诊系统两次诊断的健康值得分存储在features/O/目录下的value.txt文件中，如下图所示：
0.17753856117402078,-0.5,-0.20036608795299737,0.0,-0.5,-0.2549182256287673
-0.24459529352026832,-0.6,-0.2731810252621164,0.0,-0.6,-0.33914962777555085
后续对该用户进行诊断时，会使用到这两次诊断的历史数据，
可以对该用户的舌体多次进行诊断，看结果是否和第二次诊断的结果一致。

