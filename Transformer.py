
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import cached_dataloader
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor, ViTModel
# global variables
BATCH_SIZE = 64
TRAIN_SPLIT = 0.9
MLP_HIDDEN_SIZES = [1024,512,256]
DROPOUT_PROB = [0, 0, 0]
LR = 0.1
MOMENTUM = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(device)
train_dataset, val_dataset = cached_dataloader.getData(BATCH_SIZE, TRAIN_SPLIT)

for x in train_dataset:
    print(type(x))
    break

for x in val_dataset:
    print(type(x))
    break

class MLP(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_sizes=[128, 64], dropout_probability=[0.5,0.7]):
        super(MLP, self).__init__()
        assert len(hidden_sizes) >= 1 , "specify at least one hidden layer"
        
        self.layers = self.create_layers(in_channels, num_classes, hidden_sizes, dropout_probability)


    def create_layers(self, in_channels, num_classes, hidden_sizes, dropout_probability):
        layers = []
        layer_sizes = [in_channels] + hidden_sizes + [num_classes]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_probability[i]))
            else:
                layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        return out



class CombinedModel(nn.Module):
    def __init__(self, modality1, configuration,device):
        super().__init__()
        self.device = device
        self.modality1 = modality1.to(self.device)
        self.config = configuration

        self.head = MLP(in_channels=self._calculate_in_features(),
                            num_classes=self.config['Models']['mlp_num_classes'],
                            hidden_sizes=self.config['Models']['mlp_hidden_sizes'], 
                            dropout_probability= self.config['Models']['mlp_dropout_prob']).to(self.device)

        if(configuration['Models']['encoder_finetuning'] == False):
            for param in self.modality1.parameters():
                param.requires_grad = False

  

        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, input1):
        image_output = self.modality1(input1)['last_hidden_state'].to(self.device)
        image_output = torch.nn.Flatten()(image_output).to(self.device)
        head_output = self.head(image_output).to(self.device)
        return head_output

    def _calculate_in_features(self):
        # Create an example input and pass it through the network to get the output size
        image_list=[]
        img_batch = torch.randint(0, 255, size=(self.config['Dataset']['batch_size'], 3, 224, 224)).float()
        image_list.extend([image for image in img_batch])
        img_processor = self.config['Models']['image_processor']
        input1 = img_processor(image_list, return_tensors='pt').to(self.device) 
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.nn.Flatten()(image_output).to(self.device)
        return image_output.shape[1]
model_vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
img_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

configuration={}
configuration['Models']={}
configuration['Models']['mlp_num_classes']=2
configuration['Models']['mlp_hidden_sizes']= [1024,512,256]
configuration['Models']['mlp_dropout_prob']=[0.5,0.5,0.4]
configuration['Models']['encoder_finetuning']=True
configuration['Models']['image_processor']=img_processor
configuration['Dataset']={}
configuration['Dataset']['batch_size']=BATCH_SIZE
model=CombinedModel(model_vit,configuration,device)



from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


# Define the training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    train_f1 = 0.0
    image_list = []
    for i, (images, labels) in tqdm(enumerate(train_dataset), total = len(train_dataset), desc=f"[Epoch {epoch}]",ascii=' >='):
        image_list.extend([image for image in images])
        labels = labels.to(device)
        processed_imgs = img_processor(image_list, return_tensors='pt', data_format='channels_first').to(device)
        outputs = model(processed_imgs['pixel_values'])
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        image_list = []
        train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        train_accuracy += accuracy_score(labels.cpu(), preds.cpu())
        train_f1 += f1_score(labels.cpu(), preds.cpu(), average='macro')
        
    train_loss /= len(train_dataset)
    train_accuracy /= len(train_dataset)
    train_f1 /= len(train_dataset)
    
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_f1 = 0.0
    with torch.no_grad():
        for j, (images, labels) in tqdm(enumerate(val_dataset), total = len(val_dataset), desc=f"[Epoch {epoch}]",ascii=' >='):
            image_list.extend([image for image in images])
            labels = labels.to(device)
            processed_imgs = img_processor(image_list, return_tensors='pt', data_format='channels_first').to(device)
            outputs = model(processed_imgs['pixel_values'])
            loss = loss_function(outputs, labels)
            image_list = []
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_accuracy += accuracy_score(labels.cpu(), preds.cpu())
            val_f1 += f1_score(labels.cpu(), preds.cpu(), average='macro')
    
    val_loss /= len(val_dataset)
    val_accuracy /= len(val_dataset)
    val_f1 /= len(val_dataset)
    
    print(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f} | Train F1-score: {train_f1:.4f} | Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.4f} | Validation F1-score: {val_f1:.4f}")
# Create some sample input data
input_data = torch.randn(BATCH_SIZE,3,224,224).to(device)

# Save your PyTorch model in TorchScript format
traced_model = torch.jit.trace(model, input_data)
traced_model.save("ViTmodel.pt")