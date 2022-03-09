

STOIC2021代码结构

~~~
stoic2021:
├── algorithm # 部署时存放模型权重的地方。
├── build.bat
├── build.sh # 创建docker脚本，参见官方![README](README.md)
├── configs # openmmlab的模型配置文件
├── data # 存放数据软连接的目录，不会同步到git。
├── Dockerfile # build.sh执行时实际调用的docker蓝本。
├── export.bat
├── export.sh # 将构建好的docker容器压缩成包的脚本
├── LICENSE
├── mmdet # 模型训练和推理的核心代码
├── monai # 数据前处理，主要是3D处理的代码。
├── process.py # docker打包好之后，推理时的接口脚本。
├── __pycache__
├── README.md
├── README_MM.md # 基于mmdet的训练和推理的说明
├── requirements.txt
├── scripts # 自建的代码，其中包含了数据前处理代码， 推理测试代码和结果分析可视化等，
├── sleep_run_024.sh
├── sleep_run_135.sh
├── test
├── test.bat
├── test.sh
├── tools # 模型训练和推理的入口代码
├── utils.py
└── work_dirs # 存放模型训练结果的地方。

~~~

# data preprocess

执行代码为
~~~
./scripts/prepare4selfup.py
~~~

对肺炎数据进行批量处理，流程如下。
1. 生成Lung_mask。如果提前生成的话，直接读取。默认是直接读取。
2. 基于lung_mask对原图进行裁剪。
    基于给定的extend3axis_mm 在lung_mask基础上往每个轴测量扩展一定像素，做裁剪。
3. 根据target_spacing或者target_shape将裁剪后的图像resize到指定大小。
    target_spacing和targe_shape不应该同时制定，只提供其中一个。
4. 如果需要，可以基于肺中心做裁剪。相关代码被注释掉了。
5. 存储图像，把患者信息存放到文件名中，方便核对和使用，包括pid,age,sex,covid,severe。



数据处理后的数据存放结果如下。
~~~
10.3.4.67:/mnt/data4t/dejuns/stoic2021
├── information_v4.csv # 五折拆分信息
├── processed
│   ├── 10010_age85_sexF_covid0_severe0_lung.nii.gz # 肺分割结果
│   ├── 10010_age85_sexF_covid0_severe0.nii.gz # 原图放缩+裁剪后结果
│   ├── 10010_age85_sexF_covid0_severe0.png
│   ...
│   ├── 9994_age35_sexM_covid0_severe0_lung.nii.gz
│   ├── 9994_age35_sexM_covid0_severe0.nii.gz
│   ├── 9994_age35_sexM_covid0_severe0.png
│   ├── stoic2021_case_info.csv # 前处理后所有case的信息表格
│   └── stoic2021_case_info_lungregion.csv
├── stoic2021_case_info.csv
├── stoic2021_case_info_split.csv # 结合五折拆分信息的数据表格。
├── STOICAlgorithm_OSME.tar.gz # 发版docker 
└── STOICAlgorithm.tar.gz # 发版docker
~~~

# 模型训练

## 新冠和严重程度分类

单卡训练命令
~~~
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/stoic2021/resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode_osme.py 
~~~

多卡训练命令
~~~
CUDA_VISIBLE_DEVICES=1,3,5 PORT=28135 bash ./tools/dist_train.sh configs/stoic2021/resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode_osme.py 3
~~~


训练结果

~~~
resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode_osme
├── 20220304_110627.log # 训练log
├── 20220304_110627.log.json
├── best_severe_auc_epoch_38.pth # 训练中验证集上最好的模型，没有TTA
├── best_weight_cv05-2a1dff9a.pth # 剔除其他信息后，只包含模型权重的文件。
├── epoch_100.pth 
├── epoch_94.pth
├── epoch_96.pth
├── epoch_98.pth
├── latest.pth -> epoch_100.pth
├── resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode_osme.py # 包含所有本模型所有配置信息的文件，可用于部署
└── visual_stoic2021_test_nii # 存放TTA推理结果的目录。
    ├── aug_inference_0@4.csv # 将测试集分成四组后挨个推理的结果表格。调用脚本是 ./scripts/infer_stoic.sh
    ├── aug_inference_1@4.csv
    ├── aug_inference_2@4.csv
    └── aug_inference_3@4.csv
~~~


发布训练权重，仅保留模型权重，可节省接近一半空间。
~~~
python tools/model_converters/publish_model.py $work_dirs/$model_name/${weightfile} $work_dirs/$model_name/publish.pth
~~~


## 模型推理测试


~~~
bash scripts/infer_stoic.sh $gpuix $numfold
~~~

* 在一块GPU上，将测试数据拆分成几组($numfold)，同时进行推理，减少推理等待时间。
* infer_stoic.sh脚本中可指定模型、选用的权重文件 和数据（哪一折作为测试集）。

独立推理完成之后需要综合几组结果、计算AUC并可视化，脚本见 ./scripts/check_metric.ipynb

## 自监督预训练

单卡训练命令
~~~
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/selfup4med/simmim_convnext_s32c32em4_stoic_224x192x192_100eps_LN.py --no-validate
~~~

多卡训练命令
~~~
CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/selfup4med/simmim_convnext_s32c32em4_stoic_224x192x192_100eps_LN.py 3 --no-validate
~~~



# 模型部署

部署时，模型配置文件和存放结构如下。
~~~
git/stoic2021/algorithm/resnet/
├── best_weight_cv05.pth
├── best_weight_cv15.pth
├── best_weight_cv25.pth
├── best_weight_cv35.pth
├── best_weight_cv45.pth
└── resnset_s32c16em2_stoic2kcv25_256x224x224_2cls_agecode_osme.py
~~~

* 分类模型初始化和调用的函数参见 ./scripts/infer_clstask.py中的CovidSeverePredictor。
* Test-Time Augmentation的参数都是在这里设定的，而不会使用配置文件中的。值得注意的是，部署中的前处理都会在GPU上做，以实现加速。

~~~
class CovidSeverePredictor:
    AGE_MAP = {35: 1, 45: 2, 55: 3, 65: 4, 75: 5, 85: 6}
    SEX_MAP = {'F': 0, 'M': 1, 'A': 2, 'O': 2, 'N': 2}

    model_dir = './algorithm/resnet'
    model_name = 'resnset_s32c16em2_stoic2kcv25_256x224x224_2cls_agecode_osme'
    best_weights = ['best_weight_cv05', 
                    'best_weight_cv15', 
                    'best_weight_cv25',
                    'best_weight_cv35', 
                    'best_weight_cv45' 
                    ]
                    
    def __init__(self, model_dir = None, model_name = None, best_weights = None, 
                        device = 'cuda:0', deploy = False):

        self.device = device
        self.extend3axis_mm = (12, 12, 8) # 基于lung_mask外扩多少进行裁剪。
        self.target_spacings = None # [(1.0, 1.0, 1.0), (1.1, 1.1, 1.1)] # 是否根据target_spacings做resize。
        self.target_shapes =  [(280, 250, 250), (260, 230, 230)] # # 是否根据target_shapes做resize。
        self.flip_directions = [None, ] # 'diagonal' # 是否做图像翻转。
        self.deploy = deploy
        
        if model_dir is not None: self.model_dir = model_dir
        if model_name is not None: self.model_name = model_name
        if best_weights is not None: self.best_weights = best_weights

~~~