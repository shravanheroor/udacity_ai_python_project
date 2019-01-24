from network_model import loadCheckpoint, loadClassNames
from workspace_utils import active_session
from PIL import Image
import numpy as np
import seaborn as sns
import argparse, sys
import torch
from torch.autograd import Variable

# function to process an image
def processImage(image):
    ''' 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image)
    ar = im.width / im.height
    sz = (256, int(256/ar)) if ar < 1 else (int(ar * 256), 256) 
    im = im.resize(sz)
    center = (im.width / 2, im.height / 2)
    im = im.crop((center[0] - 112, center[1] - 112, center[0] + 112, center[1] + 112))
    
    # conver to numpy array and then normalize 
    npImage = np.array(im)
    npImage = np.divide(npImage, 255)
    npImage = np.subtract(npImage, np.array([0.485, 0.456, 0.406]))
    npImage = np.divide(npImage, np.array([0.229, 0.224, 0.225]))
    return np.transpose(npImage, (2, 0, 1))
    
def predict():
    ''' Main entry point to the train function. Sets up the arguments and runs the trainer '''
    parser = argparse.ArgumentParser(description='Predict the output.')
    parser.add_argument("imagePath", help="Specifies the image to load")
    parser.add_argument("checkpointPath", help="Specifies the checkpoint file to load")
    parser.add_argument("--top_k", help="Specifies the no of top matches to produce", type=int, default=1, dest='K')
    parser.add_argument("--category_names", help="Specifies the files for the category names", dest='catNames', default='cat_to_name.json')
    parser.add_argument("--gpu", help="Flag to enable GPU", action='store_true')
    
    args = parser.parse_args()
    
    try:
        # get the target device
        if args.gpu and not torch.cuda.is_available():
            raise Exception("GPU Not Supported")
        
        # load the device                  
        device = torch.device("cuda:0" if args.gpu else "cpu")
        
        # load checkpoint
        newmodel, classtoidx = loadCheckpoint(args.checkpointPath, device)
        
        # load the category names
        catNames = loadClassNames(args.catNames)
        
        # load the image
        testImg = processImage(args.imagePath)

        # compute the reverse map of index to class name
        idxtoclass = {v: k for k, v in classtoidx.items()}
        # run in eval mode
        newmodel.eval()
        
        # no gradients 
        with torch.no_grad():
            # convert to a batch of tensors of size 1
            pyImg = torch.from_numpy(testImg).unsqueeze_(0).to(device, dtype=torch.float)
            output = newmodel.forward(pyImg)
            # since we used the Log Softmax, use the exp to get the probabilities
            torch.exp_(output)
            probPy, indicesPy = output.topk(args.K)
            probs = probPy.cpu().numpy()
            indices = indicesPy.cpu().numpy()
            classes = [catNames[idxtoclass[idx]] for idx in indices[0]]
            return probs[0], classes
                            
    except Exception as e:
        print(e)
        raise e
        
 # module entry point
if __name__ == "__main__":
    probs, classes = predict()
    print(probs)
    print(classes)