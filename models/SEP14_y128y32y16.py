import torchvision
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets,y):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[y] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)
img_size,weights=384,'IMAGENET1K_SWAG_E2E_V1'
flag_BILINEAR=False
BATCH_SIZE=256
transform_train = v2.Compose([
    v2.Resize((img_size,img_size),interpolation=InterpolationMode.BILINEAR) if flag_BILINEAR else v2.Resize((img_size,img_size),interpolation=InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    v2.RandomHorizontalFlip(p=0.5),
    ])
transform_val = v2.Compose([
    v2.Resize((img_size,img_size),interpolation=InterpolationMode.BILINEAR) if flag_BILINEAR else v2.Resize((img_size,img_size),interpolation=InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
train_split2=ImageFolder(root="/home/yif22003/cse5095/cnfood241/train600x600")
test_split2=ImageFolder(root="/home/yif22003/cse5095/cnfood241/val600x600")
train_split = MyDataset(train_split2, transform_train)
test_split=MyDataset(test_split2, transform_val)
num_workers=20
train_ds = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True,num_workers=num_workers)
val_ds = DataLoader(test_split, batch_size=BATCH_SIZE,shuffle=False,num_workers=num_workers,drop_last=False)
n_v=len(val_ds)
n_v2=len(train_ds)
NUM_CLASSES=241
cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
flag_use_focal_loss=True
if flag_use_focal_loss:
  b=[936, 680, 739, 372, 242, 936, 936, 393, 396, 1377, 936, 936, 678, 936, 499, 936, 936, 936, 1385, 733, 936, 887, 936, 1366, 503, 732, 936, 451, 936, 207, 1204, 754, 936, 936, 755, 921, 733, 808, 936, 336, 407, 896, 215, 1383, 434, 791, 139, 368, 230, 105, 936, 1239, 813, 936, 936, 245, 1065, 441, 936, 333, 1231, 936, 1265, 1000, 489, 248, 936, 570, 936, 669, 936, 1828, 469, 936, 280, 748, 401, 936, 270, 936, 620, 936, 635, 730, 936, 280, 936, 936, 399, 936, 1325, 839, 936, 933, 936, 876, 936, 1399, 597, 691, 460, 1051, 286, 496, 529, 633, 1115, 936, 489, 91, 936, 1317, 1229, 936, 936, 257, 1391, 726, 936, 629, 504, 205, 514, 907, 322, 936, 744, 854, 140, 324, 936, 1369, 936, 327, 936, 936, 427, 936, 438, 936, 433, 700, 936, 936, 936, 936, 936, 936, 936, 936, 771, 936, 936, 916, 936, 936, 148, 279, 936, 936, 116, 936, 436, 943, 936, 936, 1235, 751, 239, 33, 1385, 936, 901, 936, 308, 443, 936, 936, 297, 808, 804, 112, 482, 621, 107, 1046, 87, 504, 1234, 936, 577, 547, 464, 936, 108, 936, 936, 936, 278, 936, 936, 936, 373, 936, 868, 936, 852, 936, 408, 400, 369, 439, 402, 191, 451, 384, 437, 411, 378, 431, 438, 371, 334, 836, 396, 440, 340, 544, 317, 413, 392, 443, 426, 445, 369, 318, 286, 429, 450, 834, 375]
  class_weights = []
  for count in b:
      weight = 1 / (count / sum(b))
      class_weights.append(weight)
  class_weights = torch.FloatTensor(class_weights).to(device)
  criterion = FocalLoss(alpha=class_weights, gamma=2).to(device)
else:criterion = nn.CrossEntropyLoss().to(device)
val_criterion=nn.CrossEntropyLoss().to(device)
with open("/home/yif22003/food_imaging/nutrient/new nutrient_label 2.txt","r") as f:
  lines=f.readlines()
t=[]
for line in lines:
  s1=line
  t1=[]
  flag=True
  while flag:
    idx1=s1.find("=")
    idx2=s1.find(",")
    if not idx1==-1:
      value=round(float(s1[idx1+1:idx2]),2)
      t1.append(value)
      s1=s1[idx2+1:]
    else:
      flag=False
  t1=np.array(t1)
  l=[2,3,4,6,7,8,9,10,12,15,16,17,18,19,23,24,25,26,27,28,29,30,31,32] # target excludes total_edible_weight, total_ingredient_weight, energy (KJ), ash, carotene, retinol, vitamin E α-E, vitamin E (β+γ)-E, vitamin E δ-E
  t1=t1[l]
  t.append(t1)
nutrient_label = np.array(t)
class GatedHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedHead, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        # _device=x.device
        # gate_values = self.sigmoid(self.gate(x)).to(_device)
        gate_values = self.sigmoid(self.gate(x))
        gated_output = gate_values * x
        # output = self.fc(self.relu(gated_output)).to(_device)
        output = self.fc(self.relu(gated_output))
        return output
class CombinedModel(nn.Module):
    def __init__(self, model1, model2, model3, gated_head):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.gated_head = gated_head
    def forward(self, x):
        # device_ = x.device
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        # combined = torch.cat((out1, out2, out3), dim=1).to(device_)
        combined = torch.cat((out1, out2, out3), dim=1)
        output = self.gated_head(combined) +out1/3 + out2/3 + out3/3
        return output
model1 =  torchvision.models.regnet_y_16gf(weights=weights)
model1.fc = torch.nn.Linear(model1.fc.in_features, NUM_CLASSES)
log_str="SEP12_y16-1"
check_point=torch.load(f"/home/yif22003/food_imaging/nutrient/{log_str}_state_best.pt")
model1.load_state_dict(check_point['model_dict'])
model1 = model1.to(device)
model2= torchvision.models.regnet_y_32gf(weights=weights)
model2.fc = torch.nn.Linear(model2.fc.in_features, NUM_CLASSES)
log_str="SEP12_y32-1"
check_point=torch.load(f"/home/yif22003/food_imaging/nutrient/{log_str}_state_best.pt")
# model2= nn.DataParallel(model2)
new_state_dict = OrderedDict()
for k, v in check_point['model_dict'].items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v
# model2.load_state_dict(check_point['model_dict'])
model2.load_state_dict(new_state_dict)
model2 = model2.to(device)
model3 =  torchvision.models.regnet_y_128gf(weights=weights)
model3.fc = torch.nn.Linear(model3.fc.in_features, NUM_CLASSES)
log_str="SEP12_y128-3"
check_point=torch.load(f"/home/yif22003/food_imaging/nutrient/{log_str}_state_best.pt")
# model3= nn.DataParallel(model3)
new_state_dict = OrderedDict()
for k, v in check_point['model_dict'].items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v
# model3.load_state_dict(check_point['model_dict'])
model3.load_state_dict(new_state_dict)
model3 = model3.to(device)
for param in model1.parameters():
    param.requires_grad = False
for param in model2.parameters():
    param.requires_grad = False
for param in model3.parameters():
    param.requires_grad = False
gated_head = GatedHead(input_dim=3*NUM_CLASSES, output_dim=NUM_CLASSES)
model = CombinedModel(model1, model2, model3, gated_head)
model= nn.DataParallel(model)
model = model.to(device)
init_lr=1e-3
optimizer = optim.Adam(model.parameters(), lr=init_lr)
finished_num_epochs=0
num_epochs=10
best_val_loss=9999
best_val_acc=0
flag_writer=0
if flag_writer:writer = SummaryWriter()
start_time=time.time()
for epoch in range(finished_num_epochs,finished_num_epochs+num_epochs):
    model.train()
    v1=0.
    qwe=0
    train_acc=0.
    train_acc_top5=0.
    for step,(x,y) in enumerate(train_ds):
        qwe+=1
        y=y.to(device)
        x=x.to(device)
        x,y2=cutmix_or_mixup(x, y)
        # print("Input tensor device:", x.device)
        if  qwe%(max(3,int(0.1*n_v2)))==0:
            print(f"epoch={epoch+1},training iter={qwe}/{n_v2}={round(100*qwe/n_v2,2)}%,device={device},elapsed_time={time.time()-start_time}")
        optimizer.zero_grad()
        output = model(x)
        if flag_use_focal_loss:
          loss = criterion(output, y2,y)
        else:loss = criterion(output, y2)
        loss.backward()
        optimizer.step()
        if flag_writer:writer.add_scalar('Memory/Allocated', torch.cuda.memory_allocated(), epoch * n_v2 + step)
        if flag_writer:writer.add_scalar('Memory/Reserved', torch.cuda.memory_reserved(), epoch * n_v2 + step)
        v_loss=loss.item()
        v1+=v_loss
        accuracy = (output.argmax(dim=1) == y).float().mean()
        train_acc+=accuracy.item()
        _,idx_top5=torch.topk(torch.nn.functional.softmax(output,dim=1),5,1)
        batch_train_acc_top5=0
        ty=-1
        for gt in y:
            ty+=1
            if gt in idx_top5[ty]:batch_train_acc_top5+=1
        batch_train_acc_top5/=len(y)
        train_acc_top5+=batch_train_acc_top5
    train_acc /= n_v2
    train_acc_top5/= n_v2
    v1 /= n_v2
    val_loss = 0.
    val_accuracy = 0.
    val_acc_top5=0.
    model.eval()
    with torch.no_grad():
        rerer=0
        for step,(x,y) in enumerate(val_ds):
            rerer+=1
            x=x.to(device)
            y=y.to(device)
            output = model(x)
            val_loss += val_criterion(output, y).item()
            val_accuracy += ((output.argmax(dim=1) == y).float().mean().item())
            _,idx_top5=torch.topk(torch.nn.functional.softmax(output,dim=1),5,1)
            batch_val_acc_top5=0
            ty=-1
            for gt in y:
                ty+=1
                if gt in idx_top5[ty]:batch_val_acc_top5+=1
            batch_val_acc_top5/=len(y)
            val_acc_top5+=batch_val_acc_top5
            if rerer%(max(1,int(0.35*n_v)))==0:
                print(f"epoch={epoch+1},validating iter={rerer}/{n_v}={round(100*rerer/n_v,2)}%,device={device},elapsed_time={time.time()-start_time}")  
    val_loss /= n_v
    val_accuracy /= n_v
    val_acc_top5/=n_v
    # scheduler.step()
    print(f"End of epoch{epoch + 1},val_accuracy={val_accuracy},lr={optimizer.param_groups[0]['lr']},train_accuracy={train_acc},train_loss={v1},val_loss:{val_loss},val_acc_top5={val_acc_top5},train_acc_top5={train_acc_top5},device={device},elapsed_time={time.time()-start_time}")
    if val_accuracy>=best_val_acc or val_loss<=best_val_loss:
      best_val_acc=val_accuracy
      best_val_loss=val_loss
      state_path=f"/home/yif22003/food_imaging/nutrient/SEP14_y128y32y16_best.pt"
      state = {
      'epoch': epoch,
      'model_dict': model.state_dict(),
      'optimizer_dict': optimizer.state_dict(),
    #   'scheduler_dict': scheduler.state_dict(),
      'criterion': criterion,
      'best_val_loss': best_val_loss,
      'best_val_acc': best_val_acc
      }
      torch.save(state,state_path)
      print(f"SEP14_y128y32y16_best saved after epoch{epoch+1} with val_acc = {best_val_acc} and val_loss={best_val_loss}")
    else:
      state_path=f"/home/yif22003/food_imaging/nutrient/SEP14_y128y32y16_relay.pt"
      state = {
      'epoch': epoch,
      'model_dict': model.state_dict(),
      'optimizer_dict': optimizer.state_dict(),
    #   'scheduler_dict': scheduler.state_dict(),
      'criterion': criterion,
      'best_val_loss': best_val_loss,
      'best_val_acc': best_val_acc
      }
      torch.save(state,state_path)
      print(f"SEP14_y128y32y16_relay saved after epoch{epoch+1} with other model best_val_acc = {best_val_acc} and best_val_loss={best_val_loss}")
if flag_writer:writer.close()