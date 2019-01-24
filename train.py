# Imports here
from network_model import loadData, initModel, saveCheckPoint
from workspace_utils import active_session
import argparse
import sys
import torch
from torch.autograd import Variable

# function to run on validation/test data
def testData(model, criterion, testloader, device):
    '''
        Test/Validate Data using the given model, criterion and 
        test data. The model will be used in eval mode and will be
        run on the specified device
    '''
    correct = 0
    total = 0
    testLoss = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            testLoss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += 1
    
    return testLoss / count, (correct / total) * 100
    
    
# Function to train data
def trainData(model, trainloader, validationloader, epochs, print_every, criterion, optimizer, device):
    '''
        Train data using the given model, criterion and Optimizer with validation for the given epochs. The training
        will be done on the spcified device
    '''
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for index, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            # reset to train mode
            model.train()
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss, accuracy = testData(model, criterion, validationloader, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss),
                  "Test Accuracy: {:.3f}".format(accuracy))

                running_loss = 0

def train():
    ''' Main entry point to the train function. Sets up the arguments and runs the trainer '''
    parser = argparse.ArgumentParser(description='Train a Network.')
    parser.add_argument("dataDir", help="Specifies the data dir")
    parser.add_argument("--save_dir", help="Specifies the dir to save checkpoints to", default=".", dest='checkpointDir')
    parser.add_argument("--arch", help="Specifies the architecture. Supported values are densenet, vgg, alexnet", choices=['densenet', 'vgg', 'alexnet'], default='densenet')
    parser.add_argument("--learning_rate", help="Specifies the learning rate", default=0.001, type=float, dest='learningRate')
    parser.add_argument("--hidden_units", help="Specifies the Hidden Units", default=500, type=int, dest='hiddenUnits')
    parser.add_argument("--epochs", help="Specifies the no of epochs to run", type=int, default=10)
    parser.add_argument("--gpu", help="Flag to enable GPU", action='store_true')
    parser.add_argument("--classes", help="No Of Op Classes", type=int, default=102)

    args = parser.parse_args()
    
    try:
        # get the target device
        if args.gpu and not torch.cuda.is_available():
            raise Exception("GPU Not Supported")
        
        # load the device                  
        device = torch.device("cuda:0" if args.gpu else "cpu")
        
        # load data
        imageDatasets, dataLoaders = loadData(args.dataDir)

        # init the model and optimizer
        model, criterion, optimizer = initModel(args.arch, args.classes, args.hiddenUnits, device, lr=args.learningRate)

        # train the data
        if args.gpu:
            with active_session():
                trainData(model, dataLoaders['train'], dataLoaders['valid'], args.epochs, 20, criterion, optimizer, device)
        else:
            trainData(model, dataLoaders['train'], dataLoaders['valid'], args.epochs, 20, criterion, optimizer, device)
                            
        # save checkpoint
        saveCheckPoint(model, imageDatasets['train'].class_to_idx, imageDatasets['train'].classes, args.epochs, args.learningRate, args.checkpointDir + '/checkpoint.pth')
                            
        # done
        sys.exit(0)
                            
    except Exception as e:
        print(e)
        sys.exit(-1)

# module entry point
if __name__ == "__main__":
    train()