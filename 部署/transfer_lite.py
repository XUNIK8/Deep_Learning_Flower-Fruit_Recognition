import tensorflow as tf

convert = tf.lite.TFLiteConverter.from_keras_model_file("lite/final_model.h5")
convert.optimizations = [tf.lite.Optimize.OPIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("lite.h5","wb").write(tflite_quant_model)
