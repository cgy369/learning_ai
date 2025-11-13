py -m pip install {package}

**init**.py에 하단에 해당 코드 넣어주면 keras가 자동완성 코드가 추가된다.

```
# Explicitly import lazy-loaded modules to support autocompletion.
# pylint: disable=g-import-not-at-top
if __import__("typing").TYPE_CHECKING:
 from keras._tf_keras import keras
 from keras._tf_keras.keras import losses
 from keras._tf_keras.keras import metrics
 from keras._tf_keras.keras import optimizers
 from keras._tf_keras.keras import initializers
# pylint: enable=g-import-not-at-top
```
