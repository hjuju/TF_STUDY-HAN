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

    model.summary()

    print(len(model.weights))           
    print(len(model.trainable_weights))