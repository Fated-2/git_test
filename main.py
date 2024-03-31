import numpy as np
import data_load
import feature_code
import model
from keras.models import clone_model
from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model
import sys
import matplotlib.pyplot as plt
cell_line = 'K562'  # 要处理的细胞系
x_forward, x_reverse, y_forward, y_reverse = data_load.load_Bi(cell_line)  # 加载DNA序列, 得到的是一维数组, 存的是string
# 将正向+反向序列通过one_hot转换成数值形式
x_forward_feature = feature_code.one_hot(x_forward)  # shape(9427, 5000, 4)
x_reverse_feature = feature_code.one_hot(x_reverse)
y_forward_feature = feature_code.one_hot(y_forward)
y_reverse_feature = feature_code.one_hot(y_reverse)
# 将文件中的数据按照","分割 , 并存储在数组genomics中 , 用于读取基因组特征数据, 一行对应一个特定的基因组特征, 列数包含描述基因组特征的标签
genomics = np.loadtxt('feature/' + cell_line + '/genomics.csv', delimiter=',')  # shape(9427, 1148)
# 定义模型的输入形状, 构建深度学习模型的参数
input_shape_x_forward = (5000, 4)
input_shape_x_reverse = (5000, 4)
input_shape_y_forward = (5000, 4)
input_shape_y_reverse = (5000, 4)
input_shape_genomics = (len(genomics[0]),)


def get_name():
    name = []
    f = open('data/' + cell_line + '/x.bed')
    for i in f.readlines():
        if i[0] != ' ':
            name.append(i.strip().split('\t')[0])  # 先去掉字符串i两端的空白, 以\t为分隔符, 返回分割后的第一个元素
    f.close()
    return name


chr_name = get_name()  # 获取染色体的名称, 一维数组chr_name的len等于文件的行数
k = 10
i = 0

# 测试集评估指标
auprc_bal = np.zeros(k)  # 大循环一共有10次, 十折交叉
acc_bal = np.zeros(k)
mcc_bal = np.zeros(k)
precision_bal = np.zeros(k)
recall_bal = np.zeros(k)
f1_bal = np.zeros(k)
# 训练参数
MAX_EPOCH = 50  # 模型训练的最大轮数, 若模型在验证集(测试集)上的性能不再提升，训练可能会提前终止。
BATCH_SIZE = 64  # 每次训练迭代中用于更新模型权重的样本数量
learning_rate = 3e-4  # 学习率为0.0003

label = np.loadtxt('data/' + cell_line + '/label.txt')
gkf = GroupKFold(n_splits=k)  # n_splits指定了交叉验证的折数(代表数据被分成几部分)
# split方法返回一个迭代器, 每次迭代产生一个训练集和测试集的索引划分
# 这个for循环的次数取决于n_splits的值, 这里是10次
for index_train, index_test in gkf.split(label, groups=chr_name):
    print('训练集长度 : ', len(index_train))  # index_train是个多个索引的数组, 用于训练的数据的下标
    print('测试集长度 : ', len(index_test))

    # Training set
    x_forward_feature_train = x_forward_feature[index_train, :, :]
    x_reverse_feature_train = x_reverse_feature[index_train, :, :]
    y_forward_feature_train = y_forward_feature[index_train, :, :]
    y_reverse_feature_train = y_reverse_feature[index_train, :, :]
    genomics_train = genomics[index_train]
    label_train = label[index_train]

    # Divide into positive and negative
    num_pos = np.sum(label_train == 1)
    num_neg = np.sum(label_train == 0)
    print(num_pos, num_neg)
    ratio = int(num_neg / num_pos)  # 这个比值通常用于描述数据集中正负样本的不平衡程度
    index_pos = np.where(label_train == 1)
    index_neg = np.where(label_train == 0)
    print('ratio:', ratio)  # 看这个比值是否要采取策略处理不平衡(1:1是理想状态, 1:2/2:1较好, 超过则出现不平衡)

    # Positive data
    x_forward_feature_train_pos = x_forward_feature_train[index_pos]
    x_reverse_feature_train_pos = x_reverse_feature_train[index_pos]
    y_forward_feature_train_pos = y_forward_feature_train[index_pos]
    y_reverse_feature_train_pos = y_reverse_feature_train[index_pos]
    genomics_train_pos = genomics_train[index_pos]
    label_train_pos = label_train[index_pos]

    # Negative data
    x_forward_feature_train_neg_total = x_forward_feature_train[index_neg]
    x_reverse_feature_train_neg_total = x_reverse_feature_train[index_neg]
    y_forward_feature_train_neg_total = y_forward_feature_train[index_neg]
    y_reverse_feature_train_neg_total = y_reverse_feature_train[index_neg]
    genomics_train_neg_total = genomics_train[index_neg]
    label_train_neg_total = label_train[index_neg]

    # Test set
    x_forward_feature_test = x_forward_feature[index_test, :, :]
    x_reverse_feature_test = x_reverse_feature[index_test, :, :]
    y_forward_feature_test = y_forward_feature[index_test, :, :]
    y_reverse_feature_test = y_reverse_feature[index_test, :, :]
    genomics_test = genomics[index_test]
    label_test = label[index_test]

    # Balanced test set  平衡测试数据集
    num = np.arange(0, len(label_test)).reshape(-1, 1)  # 将label_test数组的长度存成一个二维数组, 每行只有一个元素, 存元素的下标
    print(num.shape)
    ros = RandomUnderSampler()  # 创建一个对象, 用于解决数据集中数据不平衡问题
    num, label_test_bal = ros.fit_resample(num, label_test)  # 平衡正负样本的数量
    num = np.squeeze(num).tolist()  # 将numPy数组(多维的)转为python中的列表(一维的)
    x_forward_feature_test_bal = x_forward_feature_test[num]
    x_reverse_feature_test_bal = x_reverse_feature_test[num]
    y_forward_feature_test_bal = y_forward_feature_test[num]
    y_reverse_feature_test_bal = y_reverse_feature_test[num]
    genomics_test_bal = genomics_test[num]

    model_score = np.zeros((ratio, len(label_test)))  # 创建一个二维数组, 初始化为0
    model_score_bal = np.zeros((ratio, len(label_test_bal)))  # 平衡后的二维数组
    j = 0
    kf = KFold(n_splits=ratio, shuffle=True, random_state=0)
    # 初始化一个列表来存储每个折叠的验证损失
    val_losses = []

    # CNN卷积神经网络模型
    cnn = model.IChrom_deep(input_shape_x_forward, input_shape_x_reverse, input_shape_y_forward,
                            input_shape_y_reverse, input_shape_genomics)
    # 训练过程的编译模型, 定义了该模型在训练过程中的损失函数、优化器、评估模型(准确率为指标)
    cnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    for _, index in kf.split(label_train_neg_total):
        print('消极训练集的长度 : ', len(index))  # _表示对接收的第一个元素不感兴趣
        x_forward_feature_train_neg = x_forward_feature_train_neg_total[index]
        x_reverse_feature_train_neg = x_reverse_feature_train_neg_total[index]
        y_forward_feature_train_neg = y_forward_feature_train_neg_total[index]
        y_reverse_feature_train_neg = y_reverse_feature_train_neg_total[index]
        genomics_train_neg = genomics_train_neg_total[index]
        label_train_neg = label_train_neg_total[index]

        # 合并正负样本的数据, 正样本pos会被添加到负样本neg的行之后
        x_forward_feature_train_kf = np.concatenate((x_forward_feature_train_pos, x_forward_feature_train_neg), axis=0)
        x_reverse_feature_train_kf = np.concatenate((x_reverse_feature_train_pos, x_reverse_feature_train_neg), axis=0)
        y_forward_feature_train_kf = np.concatenate((y_forward_feature_train_pos, y_forward_feature_train_neg), axis=0)
        y_reverse_feature_train_kf = np.concatenate((y_reverse_feature_train_pos, y_reverse_feature_train_neg), axis=0)
        genomics_train_kf = np.concatenate((genomics_train_pos, genomics_train_neg), axis=0)
        label_train_kf = np.concatenate((label_train_pos, label_train_neg))

        # Divide training set and validation set 将数据分割成训练和验证数据
        x_forward_feature_train_kf, x_forward_feature_val_kf, x_reverse_feature_train_kf, x_reverse_feature_val_kf, \
        y_forward_feature_train_kf, y_forward_feature_val_kf, y_reverse_feature_train_kf, y_reverse_feature_val_kf, \
        genomics_train_kf, genomics_val_kf, label_train_kf, label_val_kf = train_test_split(x_forward_feature_train_kf,
                                                                                            x_reverse_feature_train_kf,
                                                                                            y_forward_feature_train_kf,
                                                                                            y_reverse_feature_train_kf,
                                                                                            genomics_train_kf,
                                                                                            label_train_kf,
                                                                                            test_size=0.1,
                                                                                            random_state=0)

        early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
        # # CNN卷积神经网络模型
        # cnn = model.IChrom_deep(input_shape_x_forward, input_shape_x_reverse, input_shape_y_forward,
        #                        input_shape_y_reverse, input_shape_genomics)
        # # 训练过程的编译模型, 定义了该模型在训练过程中的损失函数、优化器、评估模型(准确率为指标)
        # cnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        # cnn.compile(loss=[model.binary_focal_loss(alpha=0.5, gamma=2)], optimizer=Adam(lr=learning_rate),metrics=['accuracy'])
        # 调用fit方法开始训练, 带_val_的数据属于验证数据
        # MAX_EPOCH是遍历整个训练数据集的次数, 即梯度下降的次数
        # callbacks回调函数, 监控模型的验证损失, 并在损失不再改善时提前终止训练

        history = cnn.fit(x=[x_forward_feature_train_kf, x_reverse_feature_train_kf, y_forward_feature_train_kf,
                   y_reverse_feature_train_kf, genomics_train_kf],
                y=label_train_kf, batch_size=BATCH_SIZE,
                epochs=MAX_EPOCH, validation_data=(
                [x_forward_feature_val_kf, x_reverse_feature_val_kf, y_forward_feature_val_kf,
                 y_reverse_feature_val_kf, genomics_val_kf], label_val_kf),
                callbacks=[early_stopping_monitor])

        # 使用训练好的模型cnn对测试集进行预测, 并对预测结果进行了处理(squeeze确保结果是一维数组),
        # model_score用于存储每个测试样本的预测结果, model_score_bal用于存储平衡测试样本后的预测结果
        model_score[j] = np.squeeze(cnn.predict(
            [x_forward_feature_test, x_reverse_feature_test, y_forward_feature_test, y_reverse_feature_test, genomics_test]))
        model_score_bal[j] = np.squeeze(
            cnn.predict([x_forward_feature_test_bal, x_reverse_feature_test_bal, y_forward_feature_test_bal,
                         y_reverse_feature_test_bal, genomics_test_bal]))
        # print(model_score)
        # print(model_score_bal)
        j = j + 1

        # 从历史记录中获取验证损失
        fold_val_loss = history.history['val_loss'][-1]
        val_losses.append(fold_val_loss)

    plt.plot(val_losses, label='Kfold' + str(i) + 'Validation_Loss')
    print('损失值是: ', val_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/home/chenyu/bachengfeng/projects/IChrom-Deep-main/pictures/' + str(i) + 'fold' + '.png')
    # 处理预测结果, 将其转换为类别标签, 有助于直接比较模型预测和真实标签
    model_pro = model_score.mean(axis=0)  # mean计算平均预测概率, 也就是说他的这第i次epoch的10次j预测之间其实相互之间没有联系 , 最后之后取了这10次的平均值
    model_pro_bal = model_score_bal.mean(axis=0)
    model_class = np.around(model_pro)  # around方法将每个值四舍五入转换为类别标签(二分类中概率≥0.5时为类别 1，否则为类别 0)
    model_class_bal = np.around(model_pro_bal)

    # Evaluation 评估性能的几个性能指标
    auprc_bal[i] = average_precision_score(label_test_bal, model_pro_bal)
    acc_bal[i] = accuracy_score(label_test_bal, model_class_bal)
    mcc_bal[i] = matthews_corrcoef(label_test_bal, model_class_bal)
    precision_bal[i], recall_bal[i], f1_bal[i], _ = precision_recall_fscore_support(label_test_bal, model_class_bal,
                                                                                    average='binary')
    print('auprc_bal:', auprc_bal)
    print('acc_bal:', acc_bal)
    print('mcc_bal:', mcc_bal)
    print('precision_bal:', precision_bal)
    print('recall_bal', recall_bal)
    print('f1_bal:', f1_bal)
    print(auprc_bal[i], acc_bal[i], mcc_bal[i], precision_bal[i], recall_bal[i], f1_bal[i])
    print(len(index_test))

    i += 1

print('10-fold cross-validation')
print('auprc_bal:', auprc_bal.mean())
print('acc_bal:', acc_bal.mean())
print('mcc_bal:', mcc_bal.mean())
print('precision_bal:', precision_bal.mean())
print('recall_bal', recall_bal.mean())
print('f1_bal:', f1_bal.mean())
print(auprc_bal.mean(), acc_bal.mean(), mcc_bal.mean(), precision_bal.mean(), recall_bal.mean(), f1_bal.mean())
print('----------------------------------------------------------------')
for i in range(10):
    print(auprc_bal[i], acc_bal[i], mcc_bal[i], precision_bal[i], recall_bal[i], f1_bal[i])
