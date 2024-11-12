import tensorflow as tf

model = tf.keras.models.load_model('rsl_model_1.h5')
print(model.input_shape)
