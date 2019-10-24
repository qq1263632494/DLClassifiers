import torch
from sklearn.model_selection import train_test_split

from ClassifyTools import Classifier
from CreateDataSet import extract_plants, PlantsData
# from MakePlantsData import make_plants_data
from ResNet import get_ResNet, ResNet

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCH = 100
# make_plants_data(IMG_SIZE, PATH='3channel64sizeplants.npy')
x_data, y_data = extract_plants(PATH='3channel128sizeplants.npy', IMG_SIZE=IMG_SIZE)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
cfg = [{'channels': 32, 'num_blocks': 2, 'stride': 1},
       {'channels': 64, 'num_blocks': 2, 'stride': 2},
       {'channels': 128, 'num_blocks': 2, 'stride': 2},
       {'channels': 256, 'num_blocks': 2, 'stride': 3}]
train_set_plants = PlantsData(x=x_train, y=y_train)
test_set_plants = PlantsData(x=x_test, y=y_test)
classifier = Classifier(nn=get_ResNet(cfg=cfg, num_classes=12,
                                      x=IMG_SIZE, ipt_channel=3))
classifier.fit(train_set=train_set_plants, batch_size=BATCH_SIZE, optim='adam',
               loss_func=torch.nn.CrossEntropyLoss(), epoch=EPOCH,
               lr=0.005)
# classifier.save('model-plants-5epoch')
# classifier.load('model-plants-5epoch')
classifier.save_model_dicts('model-dicts')
classifier.load_model_dicts('model-dicts', get_ResNet(cfg=cfg, num_classes=12,
                                                      x=IMG_SIZE, ipt_channel=3))
classifier.evaluate2(X_test=x_test, Y_test=y_test, batch_size_test=BATCH_SIZE)
