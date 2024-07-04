Welcome! This project is designed specifically around training a Faster RCNN model to identify my face, although it could easily be trained on another face.

Nothing irritates me more than having no idea how to run code someone else wrote because of lack of documentation, so I'll try to be thourough!

The dataset that I collected were simply a series of selfies and photos of my own face. Then using labelme, I drew bounding boxes around what would be "me." Labelme saves the annotations as json files, and the bounding boxes are in pascal voc format (x1,y1,x2,y2). 

Once you have your annotations done you'll want to put your labels and images in separate folders in the same directory, I would make sure the corresponding image and lable file names match, so they're guarenteed to be sorted in order.

You'll also want a dataset of other faces. Otherwise, your model will not know the difference between your face and others, and assume that anyone who looks even a little like a person is you. You can get your own dataset, but for ease of use, I used the WIDER face dataset that I found on Kaggle. 

From there we artificially expand our dataset and resize our images to a more managable size using the ablumentations library, the augmentations can be customized to include all sorts of transforms, by adding or removing methods from the list being passed to A.compose(). You don't have to do this part as long as you are happy to take a couple thousand pictures of yourself and annotate each of them

After that we combine the image datasets and the label datasets, and them split them into their test/train sets, which get placed into their respective data loaders

Then we declare our model and replace the output layer with our custom one to be trained. 

After training is complete we can verify that it trained properly, and save it. This model can then be loaded into other programs and used to make predictions in those programs

