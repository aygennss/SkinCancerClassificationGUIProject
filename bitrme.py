import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical 
import tensorflow as tf
from tensorflow.keras.models import load_model


#%%



skin_df = pd.read_csv(r'C:\Users\AYGENN\OneDrive\Masaüstü\c\HAM10000_metadata.csv')

skin_df.head()
skin_df.info()

data_folder_name = r"C:/Users/AYGENN/OneDrive/Masaüstü/c/HAM10000_images_part_1/"
ext = ".jpg"
skin_df["path"] = [ data_folder_name + i + ext for i in skin_df["image_id"]]
skin_df["image"] = skin_df["path"].map( lambda x: np.asarray(Image.open(x).resize((100,75))))
skin_df["dx_idx"] = pd.Categorical(skin_df["dx"]).codes
x = np.asarray(skin_df["image"].tolist())




sns.countplot(x = 'dx', data = skin_df)
plt.xlabel('Tür', size=12)
plt.ylabel('Yogunluk', size=12)
plt.title('Türler ve Yogunluk', size=16)


bar, ax = plt.subplots(figsize = (10,10))
plt.pie(skin_df['sex'].value_counts(), labels = skin_df['sex'].value_counts().index, autopct="%.1f%%")
plt.title('Cinsiyet', size=16)



bar, ax = plt.subplots(figsize=(10,10))
sns.histplot(skin_df['age'])
plt.title('Yaş Oranı', size=16)


value = skin_df[['localization', 'sex']].value_counts().to_frame()
value.reset_index(level=[1,0 ], inplace=True)
temp = value.rename(columns = {'localization':'location', 0: 'count'})

bar, ax = plt.subplots(figsize = (12, 12))
sns.barplot(x = 'location',  y='count', hue = 'sex', data = temp)
plt.title('Cinsiyete göre hastalıgın yeri', size = 16)
plt.xlabel('hastalık', size=12)
plt.ylabel('yogunluk', size=12)
plt.xticks(rotation = 90)

# %% stardardization
x = np.asarray(skin_df["image"].tolist())
x_mean = np.mean(x)
x_std = np.std(x)
x = np.asarray(skin_df["image"].tolist())
x = (x - x_mean)/x_std


y = to_categorical(skin_df["dx_idx"], num_classes = 7)
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3, random_state=1)



#%% CNN

input_shape = (75,100,3)

#ALEXNET

model = Sequential()


model.add(Conv2D (96, kernel_size=(11,11), strides= 4,
                        padding= 'same', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'))
model.add(MaxPool2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))
model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
model.add(MaxPool2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)) 
model.add(Dropout(0.25))
model.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

model.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
model.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
model.add(MaxPool2D(pool_size=(3,3), strides= (2,2),
                              padding= 'same', data_format= None))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(4096, activation= 'relu'))
model.add(Dropout(0.25))
model.add(Dense(4096, activation= 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))
model.summary()

        
callback = tf.keras.callbacks.ModelCheckpoint(filepath='bestmodel.h5',
                                                  monitor='val_acc', mode='max',
                                                 verbose=1)

optimizer = Adam(lr = 0.001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    batch_size = 150,
                    epochs = 5,
                    callbacks=[callback])
model.save("bestmodel.h5")
model1 = load_model("bestmodel.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model dogrulugu')
plt.ylabel('dogruluk')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Ne Kadar Kayıp')
plt.ylabel('kayıp')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


loss, acc = model.evaluate(X_test, Y_test, verbose=2)

#KENDİCNNMODEL
input_shape = (75,100,3)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same", input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7,activation="softmax"))
model.summary()

callback = tf.keras.callbacks.ModelCheckpoint(filepath='bestmodel2.h5',
                                                  monitor='val_acc', mode='max',
                                                 verbose=1)

optimizer = Adam(lr = 0.001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    batch_size = 25,
                    epochs = 5,
                    callbacks=[callback])

model.save("bestmodel2.h5")
model2 = load_model("bestmodel2.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model dogrulugu')
plt.ylabel('dogruluk')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Ne Kadar Kayıp')
plt.ylabel('kayıp')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


loss, acc = model.evaluate(X_test, Y_test, verbose=2)


# %%Guı



window = tk.Tk()
window.geometry("1080x640")
window.wm_title("Deri kanseri sınıflandırma")

## global variables
img_name = ""
count = 0
img_jpg = ""

## frames
frame_left = tk.Frame(window, width = 540, height = 640, bd = "2")
frame_left.grid(row = 0, column = 0)

frame_right = tk.Frame(window, width = 540, height = 640, bd = "2")
frame_right.grid(row = 0, column = 1)

frame1 = tk.LabelFrame(frame_left, text = "Fotoğraf", width = 540, height = 500)
frame1.grid(row = 0, column = 0)

frame2 = tk.LabelFrame(frame_left, text = "Model Ve Kayıt", width = 540, height = 140)
frame2.grid(row = 1, column = 0)

frame3 = tk.LabelFrame(frame_right, text = "Sonuç", width = 270, height = 640)
frame3.grid(row = 0, column = 0)

frame4 = tk.LabelFrame(frame_right, text = "Not", width = 270, height = 640)
frame4.grid(row = 0, column = 1, padx = 10)




# frame1
def imageResize(img):
    
    basewidth = 500
    wpercent = (basewidth/float(img.size[0]))   # 1000 *1200
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize),Image.ANTIALIAS)
    return img
    
def openImage():
    
    global img_name
    global count
    global img_jpg
    
    count += 1
    if count != 1:
        messagebox.showinfo(title = "Dikkat", message = "Yalnızca Bir Fotoğraf Seçilebilir")
    else:
        img_name = filedialog.askopenfilename(initialdir = "D:\codes",title = "Dosya Seçiniz")
        
        img_jpg = img_name.split("/")[-1].split(".")[0]
        # image label
        tk.Label(frame1, text =img_jpg, bd = 3 ).pack(pady = 10)
    
        # open and show image
        img = Image.open(img_name)
        img = imageResize(img)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(frame1, image = img)
        panel.image = img
        panel.pack(padx = 15, pady = 10)
        
        
                   
menubar = tk.Menu(window)
window.config(menu = menubar)
file = tk.Menu(menubar)
menubar.add_cascade(label = "Dosya",menu = file)
file.add_command(label = "Aç", command = openImage)

# frame3
def classification():
    
    if img_name != "" and models.get() != "":
        
        # model selection
        if models.get() == "Model1":
            classification_model = model1
        else:
            classification_model = model2
        
        z = skin_df[skin_df.image_id == img_jpg]
        z = z.image.values[0].reshape(1,75,100,3)
        
        # 
        z = (z - x_mean)/x_std
        h = classification_model.predict(z)[0]
        h_index = np.argmax(h)
        predicted_cancer = list(skin_df.dx.unique())[h_index]
       
        
        
        for i in range(len(h)):
            x = 0.5
            y = (i/10)/2
            
            if i != h_index:
                tk.Label(frame3,text = str(h[i])).place(relx = x, rely = y)
            else:
                tk.Label(frame3,bg = "green",text = str(h[i])).place(relx = x, rely = y)
        
        if chvar.get() == 1:
            
            val = entry.get()
            entry.config(state = "disabled")
            path_name = val + ".txt" # result1.txt
            
            save_txt = img_name + "--" + str(predicted_cancer)
            
            text_file = open(path_name,"w")
            text_file.write(save_txt)
          
            
            val = entrya.get()
            entrya.config(state = "disabled")
            patha_name ="     "  + "\n" + val
             
            text_file.write(patha_name)
            text_file.close()
          
        else:
            print("Kaydetme seçilmedi")
    else:
        messagebox.showinfo(title = "Hata", message = "Fortoğraf ve model seçin!")
        tk.Label(frame3, text = "Fortoğraf ve model seçin!" ).place(relx = 0.1, rely = 0.6)
                          
columns = ["lesion_id","image_id","dx","dx_type","age","sex","localization"]

            

classify_button = tk.Button(frame3, bg = "red", bd = 4, font = ("Times",13),activebackground = "orange",text = "Sonuç",command = classification)
classify_button.place(relx = 0.1, rely = 0.5)
# frame 4
labels = skin_df.dx.unique()

entrya = tk.Entry(frame4, width=55)
entrya.insert(string = "Eklenecek Notlar",index = 0)
entrya.grid(row = 1, column =1 )

for i in range(len(columns)):
    x = 0.1
    y = (i/10)/2
    tk.Label(frame3, font = ("Times",12), text = str(labels[i]) + ": ").place(relx = x, rely = y)
# frame 2 
# combo box
model_selection_label = tk.Label(frame2, text = "Sınıflandırma modeli: ")
model_selection_label.grid(row = 0, column = 0, padx = 5)

models = tk.StringVar()
model_selection = ttk.Combobox(frame2,textvariable = models, values = ("Model1","Model2"), state = "readonly")
model_selection.grid(row = 0, column = 1, padx = 5)

# check box
chvar = tk.IntVar()
chvar.set(0)
xbox = tk.Checkbutton(frame2, text = "Sonuçalrı Kaydet", variable = chvar)
xbox.grid(row = 1, column =0 , pady = 5)

# entry
entry = tk.Entry(frame2, width = 23)
entry.insert(string = "Dosya Adı",index = 0)
entry.grid(row = 1, column =1 )



window.mainloop()

































