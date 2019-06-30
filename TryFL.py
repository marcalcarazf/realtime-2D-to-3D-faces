'''
Created on Oct 17, 2018
Modified on Abr 25, 2019

@author: deckyal
@modified by: marc alcaraz
'''

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
from FacialDataset import ImageDatasets

######################## Main Methods ################################################
def initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):
    model_ft = None
    input_size = 0

    if model_name == 'resnet' :
        model_ft = models.resnet152(pretrained = use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'alexnet' :
        model_ft = models.alexnet(pretrained = use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'squeezenet' :
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        model_ft.classifier[1] = nn.Conv2d(512,num_classes, kernel_size=(1,1), stride = (1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == 'densenet' :

        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    elif model_name == 'inception' :
        #inception v3
        model_ft = models.inception_v3(pretrained = use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        #Aux net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        #Primary Net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 229

    else :
        print('invalid model name')
        exit()

    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs = 25, is_inception = False):
    print("Training model")
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 99999
    
    for epoch in range(num_epochs) : 
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
        
        running_loss = 0
        
        #For rinputs, rlabels in dataloaders :
        for x,(rinputs, rlabels) in enumerate(dataloaders,0) : 
            
            model.train()
                
            inputs = rinputs.to(device) 
            labels = rlabels.to(device)
            
            #zero the parameter gradients 
            optimizer.zero_grad()
            
            #Forward 
            #Track history if only in train 
            with torch.set_grad_enabled(True) : 
                if is_inception : 
                    outputs, aux_outputs  = model(inputs)
                    loss1 = criterion(outputs,labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1+.4 * loss2
                else : 
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
             
                loss.backward()
                optimizer.step()
                    
            #Statistics
            running_loss += loss.item() * inputs.size(0)
            print("{}/{} loss : {}".format(x,int(len(dataloader.dataset)/batch_size),loss.item()))
        
        print('Dataset len {}'.format(len(dataloaders.dataset)))
        print('Running_loss len {}'.format(running_loss))
        epoch_loss = running_loss / len(dataloaders.dataset)
        print('Loss : {:.4f}'.format(epoch_loss))
        
        #Deep copy the model 
        if epoch_loss < lowest_loss : 
            lowest_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),'netFL.pt')
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(lowest_loss))

    #Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting : 
        for param in model.parameters() : 
            param.requires_grad = False
######################################################################################



###################### Main Steps ###################################################
#Print PyTorch and Torchvision Versions and Check Cuda
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print("Cuda: ", torch.cuda.is_available())

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Main parameters
model_name = 'resnet' #[resnet, alexnet, vgg, squeezenet, densenet,inception]
num_classes = 25
batch_size = 6
num_epochs = 100
feature_extract = False

#Intialize the model for this run 
model_ft , image_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


#Data augmentation and normalization for training and just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size,image_size)),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


#Build Dataset and Dataloader
ID = ImageDatasets(data_list = ['300W3D-Train'],blurLevel=0,transform=data_transforms['val'])
dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = False)
print("Len dataset: ", len(ID))


#Params to learn
#model_ft.load_state_dict(torch.load('./netFL.pt'))
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
print("Params to learn ")

if feature_extract :
    params_to_update = []
    for name,param in model_ft.named_parameters(): 
        if param.requires_grad == True : 
            params_to_update.append(param)
            print("\t",name)
else : 
    for name,param in model_ft.named_parameters() : 
        if param.requires_grad == True :
            print("--\t",name)


#Optimizer and Loss Function
optimizer_ft = optim.SGD(params_to_update,lr=.01, momentum = .9)
criterion = nn.MSELoss()

#Train and evaluate 
print ("Start Model Training")
model_ft, hist = train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs, is_inception = (model_name == "inception"))

print("Training Ended")
#############################################################################
