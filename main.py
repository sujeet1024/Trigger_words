import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import numpy as np
#from torchvision.io import read_image


class CustomAudioDataset(Dataset):
    def __init__(self, annotations_file, x_dir, y_dir, transform=None, target_transform=None):
        self.aud_labels = annotations_file #load csv file containing training examples
        self.x_dir = x_dir #path to X_train
        self.y_dir = y_dir #path to Y_train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.aud_labels)

    def __getitem__(self, idx):
        aud_path = os.path.join(self.x_dir, self.aud_labels.iloc[idx, 0])
        audio = torch.from_numpy(np.load(aud_path))
        label_path = os.path.join(self.y_dir, self.aud_labels.iloc[idx, 1])
        label = torch.squeeze(torch.from_numpy(np.load(label_path)))
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
        return audio, label



import pandas as pd
annotations_file = pd.read_csv('./train_list.csv')
test_annotations = pd.read_csv('./test_list.csv')
x_dir = 'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/X_train'
y_dir = 'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/Y_train'
x_test_dir = 'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/X_test'
y_test_dir = 'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/Y_test'

Train_data = CustomAudioDataset(annotations_file, x_dir, y_dir)
Test_data = CustomAudioDataset(test_annotations, x_test_dir, y_test_dir)




from torch.utils.data import DataLoader

train_dataloader = DataLoader(Train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(Test_data, batch_size=64, shuffle=True)




import torch.nn.functional as F
class liftModel(torch.nn.Module):
    def __init__(self):
        super(liftModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=101, out_channels=1, kernel_size=3, stride=2)
        self.bnorm1d = torch.nn.BatchNorm1d(1)
        self.gru1 = torch.nn.GRU(input_size= 238, hidden_size=196, dropout=0.5, batch_first=True)
        self.gru2 = torch.nn.GRU(input_size= 196, hidden_size=128, num_layers=1, dropout=0.5, batch_first=True)
        self.bnorm1d2 = torch.nn.BatchNorm1d(1)
        self.ll = torch.nn.Linear(in_features=128, out_features=128, bias=True)
        self.relu = torch.nn.ReLU()
#         self.tdd = torch.nn.Conv2d(1, num_of_output_channels=1 , (num_of_input_channels=128, 1))
#         self.gru2 = torch.nn.GRU(input_size= 128*0.7, hidden_side=128, num_layers=1, dropout=0.7)
                 
    def forward(self, inp):
        out = self.conv1(inp)
        out = self.bnorm1d(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.5, training=True, inplace=False)
        out, _ = self.gru1(out)
        out = self.bnorm1d2(out)
        out, _ = self.gru2(out)
        out = self.bnorm1d2(out)
        out = F.dropout(out, p=0.5, training=True, inplace=False)
        out = self.ll(out)
        #time distributed layer
        return out




leftmodel = liftModel()
# leftmodel.cuda()




from torch.optim import Adam

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(leftmodel.parameters(), lr=0.001, weight_decay=0.0001)




nexamples = len(Train_data)
n_iterations = nexamples//64 + 1

from torch.autograd import Variable
from tqdm import tqdm


# def saveModel():
#     path = 'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/Models/modeltrial.pth'
#     torch.save(leftmodel.state_dict(), path)

def testAccuracy():
    leftmodel.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for (x, y) in test_dataloader:
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = torch.device('cpu')
            x.to(device)
            y.to(device)
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
#             y = y.type(torch.LongTensor)
            outputs = leftmodel(x)
            predicted = torch.argmax(outputs)
            print('prediction, y')
            print(x.shape)
            print(outputs.shape)
            print(predicted.shape)
            print(y.shape)
            print(predicted)
            print(y)
            total+=y.size(0)
            accuracy+= (predicted==y).sum().item()
    accuracy = (100.*accuracy/total)    # compute accuracy over all test images
    return accuracy


def train(epochs):
    best_accuracy = 0.0
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print("The model will be running on ", device, " device")
    # accuracy = testAccuracy()
    with open('log.txt', 'w') as logfile:
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            # running_acc = 0.0
            with tqdm(train_dataloader, leave=False) as tepoch:    
                # for i, (x, y) in enumerate(train_dataloader):
                for (x, y) in (tepoch):
                    tepoch.set_description(f'Epoch {epoch+1}')
                    leftmodel.train()
                    #forward, backward, weigts update
                    x = torch.transpose(x.type(torch.cuda.FloatTensor), 2, 1)
                    y = y.type(torch.cuda.FloatTensor)
                    y = y.type(torch.LongTensor) # avoid RuntimeError: expected scalar type Long but found Float
                    x = Variable(x.to(device))   # can be also , dtype=torch.float   or can transfer model to cuda
                    y = Variable(y.to(device))   # can be also , dtype=torch.float
                    optimizer.zero_grad()   # zero the parameter gradients
                    output = leftmodel(x)   # predict o/p
                    loss = loss_fn(output, y)  # compute loss
                    loss.backward()         # backpropagate the loss
                    optimizer.step()        # adjust params based on calculated grads
                    running_loss+=loss.item()  # extract loss value
                    
                    # accuracy = testAccuracy()
                    # tepoch.set_postfix(loss=running_loss/(i+1), accuracy=100.*accuracy)
                    # if (i+1) % 15 == 1:
                    #     logfile.write(f"for epoch {epoch+1}; iteration {i+1} || the loss is {loss.item()} |&| the Accuracy is {accuracy*100.}\n")
                accuracy = testAccuracy()
                tepoch.set_postfix(loss=running_loss/479., accuracy = accuracy*100.)
                logfile.write(f'for epoch {epoch+1} || the loss is {running_loss/479.} |&| the accuracy is {accuracy*100.}\n')
                #     # print(f'epoch {epoch+1}/{epochs}, step {i+1}/{n_iterations}')
                #     # print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/140))
                #     # running_loss=0.0    # zero the loss
                #     accuracy = testAccuracy()
                #     tepoch.set_postfix(loss=running_loss/140., accuracy=100.*accuracy)
                    # print('For epoch ', epoch+1, ' the test accuracy over the whole test set %d %%' % (accuracy))
                





if __name__ == "__main__":
    epochs = 1
    train(epochs)
                