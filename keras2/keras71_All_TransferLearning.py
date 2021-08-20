# pre-trained model

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7


list = [
    VGG16(), VGG19(), Xception(), 
    DenseNet121(), DenseNet169(), DenseNet201(),
    ResNet50(), ResNet50V2(),
    ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
    InceptionV3(), InceptionResNetV2(),
    MobileNet(), MobileNetV2(), MobileNetV3Large(), MobileNetV3Small(),
    NASNetLarge(), NASNetMobile(),
    EfficientNetB0(), EfficientNetB1(), EfficientNetB7()]


for i in list:
    
    model = i
   
    model.trainable = False

    # model.summary()

    print(i)
    print('Total params : ', model.count_params())
    print('전체가중치 :', len(model.weights))           
    print('훈련가능한 가중치 :', len(model.trainable_weights))

    '''
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F628542AC0>
Total params :  138357544
전체가중치 : 32
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F62C043610>
Total params :  143667240
전체가중치 : 38
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F6324DB8E0>
Total params :  22910480
전체가중치 : 236
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F6324DB3D0>
Total params :  8062504
전체가중치 : 606
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F6547B2100>
Total params :  14307880
전체가중치 : 846
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F66007D190>
Total params :  20242984
전체가중치 : 1006
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F660293340>
Total params :  25636712
전체가중치 : 320
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F6602937F0>
Total params :  25613800
전체가중치 : 272
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F660498190>
Total params :  44707176
전체가중치 : 626
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F67E054D90>
Total params :  44675560
전체가중치 : 544
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F67E0540D0>
Total params :  60419944
전체가중치 : 932
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F8299CCA30>
Total params :  60380648
전체가중치 : 816
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F829D257F0>
Total params :  23851784
전체가중치 : 378
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F8299CCF10>
Total params :  55873736
전체가중치 : 898
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82A591BE0>
Total params :  4253864
전체가중치 : 137
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82A72ADF0>
Total params :  3538984
전체가중치 : 262
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82AB32D60>
Total params :  5507432
전체가중치 : 266
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82A90B7C0>
Total params :  2554968
전체가중치 : 210
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82ADACD60>
Total params :  88949818
전체가중치 : 1546
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82B940820>
Total params :  5326716
전체가중치 : 1126
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82C121F70>
Total params :  5330571
전체가중치 : 314
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82C129AC0>
Total params :  7856239
전체가중치 : 442
훈련가능한 가중치 : 0
<tensorflow.python.keras.engine.functional.Functional object at 0x000001F82F655C40>
Total params :  66658687
전체가중치 : 1040
훈련가능한 가중치 : 0
    '''