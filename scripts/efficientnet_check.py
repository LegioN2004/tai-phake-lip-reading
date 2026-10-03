print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

eff = keras.applications.EfficientNetB0(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

for layer in eff.layers[:5]:
    print(layer.name, layer.__class__.__name__)