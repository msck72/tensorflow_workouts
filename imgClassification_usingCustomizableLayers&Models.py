import tensorflow as tf

(x_train, y_train) , (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# x_train = tf.reshape(x_train, (-1 , 28 * 28))
# x_test = tf.reshape(x_test, (-1 , 28 * 28))

print(x_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

print(len(x_train))


class CustomizedLayer(tf.keras.layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super().__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape = (input_dim, units), dtype='float32'),
            trainable = True
        )

        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(
            initial_value = b_init(shape = (units , ), dtype='float32'),
            trainable = True
        )

    def call(self, inputs):
        return tf.matmul(inputs , self.w) + self.b



class CustomizedModel(tf.keras.Model):
    def __init__(self, input_dimensions , layers_dim):
        super().__init__()
        # self.layer_1 = CustomizedLayer(units =  1024, input_dim = 28 * 28)
        self.myLayers = []
        for i in range(len(layers_dim)):
            # print(f"output_dim = {layers_dim[i]}, input_dim = {input_dimensions}")
            self.myLayers.append(CustomizedLayer(units = layers_dim[i], input_dim = input_dimensions))
            input_dimensions = layers_dim[i]
        
        # self.layer_1 = CustomizedLayer(units = layer, input_dim = input_dimensions)
        # self.layer_2 = CustomizedLayer()

    def call(self , input):
        input = tf.keras.layers.Flatten()(input)
        # print("in MODEL") 
        # print(f"input_shape = {input.shape}")
        x = self.myLayers[0](input)
        # print(f"input_shape = {input.shape} output_shape = {x.shape}")
        i = 1
        while i < len(self.myLayers):
            # print(f"input_shape = {x.shape}")
            x = self.myLayers[i](x)
            # print(f"output_shape = {x.shape}")
            if i != len(self.myLayers) - 1 :
                x = tf.nn.relu(x)
            i += 1
        return x

@tf.function
def passImageIntoModel(image , label):
    with tf.GradientTape() as tape:
        prediction = model(input = image)
        loss = myloss_func(label , prediction)
    myGradient = tape.gradient(loss , model.trainable_variables)
    optimizer.apply_gradients(zip(myGradient , model.trainable_variables))

    # note the loss and accuracy
    train_loss(loss)
    train_accuracy(label, prediction)

@tf.function
def test_passImageIntoModel(image , label):
    with tf.GradientTape() as tape:
        prediction = model(test_image)
        loss = myloss_func(test_label , prediction)
    # note the loss and accuracy
    test_loss(loss)
    test_accuracy(test_label, prediction)


    
model = CustomizedModel(len(x_train[0]) * len(x_train[1]), [1024 , 512 , 256 , 64 , 10])

myloss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


EPOCHS = 5

for epochs in range(EPOCHS):
    
    train_loss.reset_states()
    train_accuracy.reset_states()

    for image , label in train_dataset:
        passImageIntoModel(image , label)

    print(f"Epoch : {epochs + 1} , loss = {train_loss.result()} , accuracy = {train_accuracy.result() * 100}")
# evaluate the model by testing

for test_image, test_label in test_dataset:
    test_passImageIntoModel(test_image, test_label)
   


print(f"test_loss = {test_loss.result()} , test_accuracy = {test_accuracy.result() * 100}")