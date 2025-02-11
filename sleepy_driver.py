# In[1]:


import os 
import numpy as np 
import matplotlib.pyplot as plt 

import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential 


# In[2]:


path="C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver"


# In[3]:


path


# In[4]:


os.listdir(path)


# In[5]:


os.listdir("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\validation")


# In[6]:


os.listdir("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\Train")


# In[7]:


len(os.listdir("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\Train\\opened"))


# In[8]:


len(os.listdir("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\Train\\closed"))


# In[9]:


len(os.listdir("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\validation\\closed"))


# In[10]:


len(os.listdir("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\validation\\opened"))


# In[11]:


os.listdir("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\Train\\opened")


# In[12]:


import PIL


# In[13]:


PIL.Image.open("C:\\Users\\ASUS TUF F15\\Downloads\\Sleepy_driver\\Train\\opened\\1004.jpg")


# In[14]:


train_dir=os.path.join(path,'train')


# In[15]:


train_dir


# In[16]:


val_dir=os.path.join(path,'validation')
val_dir


# In[17]:


batch_size=64
img_size=(180,180)


# In[18]:


train_ds=tf.keras.utils.image_dataset_from_directory(train_dir,
                                                    shuffle=True,
                                                    batch_size=batch_size, 
                                                    image_size=img_size)


# In[19]:


val_ds=tf.keras.utils.image_dataset_from_directory(val_dir, 
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    image_size=img_size)


# In[20]:


class_names=train_ds.class_names
class_names


# In[21]:


for images,labels in train_ds.take(1):
    print(images[0].numpy().astype('uint8'))
    print(labels)


# In[22]:


plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1) 
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        


# In[23]:


# splitting validation data into test data and validation data 


# In[24]:


val_batches=tf.data.experimental.cardinality(val_ds)


# In[25]:


test_ds=val_ds.take(val_batches//5)
val_ds=val_ds.skip(val_batches//5)


# In[26]:


print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))


# In[27]:


AUTOTUNE=tf.data.AUTOTUNE
train_ds=train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds=test_ds.prefetch(buffer_size=AUTOTUNE)


# In[28]:


data_augmentation=Sequential([
    tf.keras.layers.RandomFlip('vertical',input_shape=(180,180,3)), 
    tf.keras.layers.RandomRotation(0.3)
])


# In[29]:


for image, _ in train_ds.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image,0))
    plt.imshow(augmented_image[0] / 255.0)
    plt.axis('off')


# In[30]:


image_batch,label_batch=next(iter(train_ds))
image_batch.shape


# In[31]:


label_batch


# In[32]:


model=Sequential([
    data_augmentation, 
    layers.Rescaling(1./255), 
    layers.Conv2D(8,3,activation='relu'), 
    layers.MaxPool2D(), 
    layers.Conv2D(16,3,activation='relu'), 
    layers.MaxPool2D(), 
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPool2D(), 
    layers.Flatten(),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])


# In[33]:


model.summary()


# In[34]:


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])


# In[35]:


initial_epochs=30
loss0,accuracy0=model.evaluate(val_ds)


# In[36]:


history=model.fit(train_ds,
                 epochs=initial_epochs, 
                 validation_data=val_ds)


# In[37]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(initial_epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[38]:


model.evaluate(val_ds)


# In[39]:


#evaluating test dataset
test_losss,test_accuraccy=model.evaluate(test_ds)


# In[40]:


test_losss


# In[41]:


test_accuraccy


# In[42]:


from tensorflow.keras.utils import img_to_array,load_img


# In[43]:


img=load_img('opens.jpg',target_size=(180,180))


# In[44]:


img


# In[45]:


img_arr=img_to_array(img)


# In[46]:


img_arr


# In[47]:


pred1=model.predict(img_arr.reshape(1,180,180,3))[0]


# In[48]:


pred1


# In[49]:


if pred1[0] > 0.5:  
    predicted_class = 1   
else:
    predicted_class = 0
print(predicted_class)


# In[50]:


class_names[predicted_class]


# In[51]:


class_names


# In[52]:


class_names[1]


# In[63]:


class_names[0]


# In[111]:


ci=load_img('close.jpg',target_size=(180,180))


# In[112]:


ci


# In[113]:


ci_ar=img_to_array(ci)


# In[114]:


ci_ar


# In[115]:


pred2=model.predict(ci_ar.reshape(1,180,180,3))


# In[116]:


pred2


# In[117]:


if pred2[0] > 0.5:  # assuming a binary classification with threshold 0.5
    predicted_class = 1   
else:
    predicted_class = 0
print(predicted_class)


# In[118]:


class_names[predicted_class]


# In[ ]:


model.save('my_model.h5')

