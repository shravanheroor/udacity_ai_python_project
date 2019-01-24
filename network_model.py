# Imports
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.autograd import Variable
import json

# Set the Model to require gradient or to freeze params
def setModelGrad(model, requireGrad=False):
    for param in model.parameters():
        param.requires_grad = requireGrad

# Initialize model and classifier and move to specified device
def initModel(name, opClasses, hiddenLayerSize, device, preTrained=True, requireGrad=False, dropout=0.5, lr=0.001):
    '''
        Based on the name, get a TV model , pretrained if required. Since we will replace the classifier, 
        we need to figure out the classifier ip layer size and this varies per model. Then create the classifier, 
        select the criterion and optimizer. Returns a 3 value tuple (model, criterion, optimizer)
    '''
    classifier = None
    model = None
    classifierIpSize = 1024
    
    # Based on the model we choose, we have to change the classifier ip size
    if name == 'densenet':
        model = models.densenet121(preTrained)
    elif name == 'vgg':
        model = models.vgg13(preTrained)
        classifierIpSize = model.classifier[0].in_features    
    elif name == 'alexnet':
        model = models.alexnet(preTrained)
        classifierIpSize = model.classifier[1].in_features
    else:
        print("Model {0} is not supported. Please Try densenet, vgg, alexnet".format(name))
        raise NotImplementedError
        
    if not model:
        raise Exception("Could not Load Model {0}".format(name))
    
    
    # create the classifier with the required layers
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifierIpSize, hiddenLayerSize)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc3', nn.Linear(hiddenLayerSize, opClasses)),
        ('op', nn.LogSoftmax(dim=1)),
    ]))
    
    # set if gradient is required on the model
    setModelGrad(model, requireGrad)
    
    # move to device
    model.to(device)

    # choose the NLLLoss since we are using a LogSoftMax as the final output of the classifier
    criterion = nn.NLLLoss()
    
    # move classifier to device and set on the model
    criterion.to(device)
    classifier.to(device)
    
    # set the classifier on the model
    model.classifier = classifier

    # use the adam optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    return model, criterion, optimizer

# Function to load Data
def loadData(dataDir):
    '''
        Loads the data from the specified dir. Expects to have 
        folder structure as 
            dataDir/train
            dataDir/test
            dataDir/valid
        Will load all the transforms and return a dictionary of dataSets and dataLoaders
    '''
    train_dir = dataDir + '/train'
    valid_dir = dataDir + '/valid'
    test_dir = dataDir + '/test'

    data_transforms_train = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.25),
        transforms.RandomVerticalFlip(0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_validation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    imageDatasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms_train),
        'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms_validation),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms_test)
    }

    dataLoaders = { 
        'train' : torch.utils.data.DataLoader(imageDatasets['train'], batch_size=64, shuffle=True),
        'valid' : torch.utils.data.DataLoader(imageDatasets['valid'], batch_size=32, shuffle=True),
        'test' : torch.utils.data.DataLoader(imageDatasets['test'], batch_size=32)
    }

    return imageDatasets, dataLoaders
   
# given a file with a map of the class to a name, load and return
def loadClassNames(fileName):
    with open('cat_to_name.json', 'r') as f:
        classNamesMap = json.load(f)
    return classNamesMap

def saveCheckPoint(currentModel, classtoidx, classes, epochs, lr, checkpointFile):
    '''
        Save some information in the checkpoint file. Most important info
        is the model, modelstate, and class to index map. We store additional 
        data
    '''
    checkpoint = {
        'epochs': epochs,
        'classtoidx': classtoidx,
        'classes' : classes,
        'learrate': lr,
        'optimizer': 'Adam',
        'model': currentModel,
        'stateDict': currentModel.state_dict()
    }

    torch.save(checkpoint, checkpointFile)

# load check point
def loadCheckpoint(checkpointFile):
    '''
        Load the checkpoint file. Return the model (with state reloaded)
        and class to index map. Additional info not returned currently
    '''
    chkpoint = torch.load(checkpointFile)
    model = chkpoint['model']
    epochs = chkpoint['epochs']
    lr = chkpoint['learrate']
    classtoidx = chkpoint['classtoidx']
    model.load_state_dict(chkpoint['stateDict'])
    return model, classtoidx
