import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization, add, Flatten, Input
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.applications.vgg19 import VGG19
from keras.models import Model
import cv2
import os
from tqdm import tqdm



def build_vgg(hr_shape):
    Vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)

    return Model(inputs=Vgg.inputs, outputs=Vgg.layers[10].output)


def res_block(ip):
    res_model = Conv2D(64, (3, 3), padding='same')(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1, 2])(res_model)
    res_model = Conv2D(64, (3, 3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)

    return add([ip, res_model])


def upscale_block(ip):
    up_model = Conv2D(256, (3, 3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1, 2])(up_model)

    return up_model

def create_gen(gen_ip, n_res_block):
    layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1, 2])(layers)
    temp = layers
    for i in range(n_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3, 3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers, temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9, 9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)


def discriminator_block(ip, filters,strides=1, bn=True):
    discriminator_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)
    if bn:
        discriminator_model = BatchNormalization(momentum=0.8)(discriminator_model)

    discriminator_model = LeakyReLU(alpha=0.2)(discriminator_model)

    return discriminator_model


def create_disc(disc_ip):
    df = 64
    d1 = discriminator_block(disc_ip, df, strides=1, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df * 2)
    d4 = discriminator_block(d3, df * 2, strides=2)
    d5 = discriminator_block(d4, df * 4)
    d6 = discriminator_block(d5, df * 4, strides=2)
    d7 = discriminator_block(d6, df * 8)
    d8 = discriminator_block(d7, df * 8, strides=2)

    d8_5 = Flatten()(d8)
    d9 = Dense(df * 16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)

def create_comb(gen_model, disc_model,vgg, lr_ip, hr_ip):
    gen_img= gen_model(lr_ip)
    gen_features= vgg(gen_img)
    disc_model.trainable= False
    validity = disc_model(gen_img)

    return Model(inputs= [lr_ip, hr_ip], outputs= [validity, gen_features])


hr_imgs = []
lr_imgs  =[]

pth_hr = 'DIV2K_train_HR/'
pth_lr = 'LR Images/'
for i in os.listdir(pth_hr):
    img1 = cv2.imread(pth_hr+i)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (128,128))
    hr_imgs.append(img1)

for j in os.listdir(pth_hr):
    img2 = cv2.imread(pth_lr+j)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, ((32,32)))
    lr_imgs.append(img2)

hr_imgs = np.array(hr_imgs)
lr_imgs = np.array(lr_imgs)

print(hr_imgs.shape)

hr_imgs = hr_imgs/255
lr_imgs = lr_imgs/255

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_imgs, hr_imgs, test_size=0.10, random_state=10)

hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

hr_ip = Input(shape=hr_shape)
lr_ip = Input(shape=lr_shape)

generator = create_gen(lr_ip, n_res_block=16)
generator.summary()

discriminator = create_disc(hr_ip)
discriminator.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
discriminator.summary()

vgg = build_vgg((128, 128, 3))
print(vgg.summary())
vgg.trainable = False

gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
gan_model.compile(loss= ['binary_crossentropy','mse'], loss_weights= [1e-3, 1], optimizer= 'adam')
gan_model.summary()


batch_size = 1
train_hr_batches=[]
train_lr_batches=[]
for it in range(int(hr_train.shape[0]/batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])

epochs= 20

for e in range(epochs):
    fake_lable = np.zeros((batch_size, 1))
    real_lable = np.ones((batch_size, 1))

    g_losses = []
    d_losses = []

    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]

        fake_imgs = generator.predict_on_batch(lr_imgs)

        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_lable)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_lable)

        discriminator.trainable = False
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

        img_features = vgg.predict(hr_imgs)

        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_lable, img_features])

        d_losses.append(d_loss)
        g_losses.append(g_loss)

    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)

    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)

    print("epoch", e + 1, "g_loss", g_loss, "d_loss", d_loss)

    if (e + 1) % 10 == 0:
        generator.save("gen_" + str(e + 1) + ".h5")


from numpy.random import randint
from keras.models import load_model
import matplotlib.pyplot as plt

g= load_model('gen_20.h5', compile=False)

[X1, X2] = [lr_test, hr_test]
x= randint(0, len(X1), 1)

scr_img, tar_img = X1[x], X2[x]

gen = g.predict(scr_img)

plt.figure(figsize=(16,8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(scr_img[0,:,:,:])
plt.subplot(232)
plt.title('SR Image')
plt.imshow(gen[0,:,:,:])
plt.subplot(233)
plt.title('HR Image')
plt.imshow(tar_img[0,:,:,:])

plt.show()