import torchvision
log_str="SEP12_y32-1"
flag_relay=True
flag_multiple_gpus=True
model_str= "y32"
img_size,weights=384,'IMAGENET1K_SWAG_E2E_V1'
model =  torchvision.models.regnet_y_32gf(weights=weights)
BATCH_SIZE = 64
init_lr=1e-5
num_epochs=T_max=30
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.linear_model import RANSACRegressor
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(1)
torch.manual_seed(1)
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
flag_BILINEAR=False    
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
val_ds = DataLoader(test_split, batch_size=BATCH_SIZE,shuffle=False,num_workers=num_workers,drop_last=True)
n_v=len(val_ds)
n_v2=len(train_ds)
NUM_CLASSES=241
cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
ransac=RANSACRegressor(min_samples=2)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
weight_decay=1e-4
optimizer = optim.AdamW(model.parameters(), lr=init_lr,weight_decay=weight_decay)
scheduler_str="CosineAnnealingLR"
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
if flag_multiple_gpus:
  model= nn.DataParallel(model)
if flag_relay:
  check_point=torch.load(f"/home/yif22003/food_imaging/nutrient/{log_str}_state_relay.pt")
  model.load_state_dict(check_point['model_dict'])
  model=model.to(device)
  optimizer.load_state_dict(check_point['optimizer_dict'])
  scheduler.load_state_dict(check_point['scheduler_dict'])
else:
  model=model.to(device)
flag_use_focal_loss=True
with open(f"/home/yif22003/food_imaging/nutrient/{log_str}.txt","a") as ff:
  ff.write(f"scheduler={scheduler_str},weights={weights},flag_multiple_gpus={flag_multiple_gpus},flag_relay={flag_relay},img_size={img_size},BATCH_SIZE={BATCH_SIZE},flag_use_focal_loss={flag_use_focal_loss},init_lr={init_lr},weight_decay={weight_decay},T_max={T_max},flag_BILINEAR={flag_BILINEAR},num_epochs={num_epochs},num_workers={num_workers}\n")
if flag_use_focal_loss:
  b=[936, 680, 739, 372, 242, 936, 936, 393, 396, 1377, 936, 936, 678, 936, 499, 936, 936, 936, 1385, 733, 936, 887, 936, 1366, 503, 732, 936, 451, 936, 207, 1204, 754, 936, 936, 755, 921, 733, 808, 936, 336, 407, 896, 215, 1383, 434, 791, 139, 368, 230, 105, 936, 1239, 813, 936, 936, 245, 1065, 441, 936, 333, 1231, 936, 1265, 1000, 489, 248, 936, 570, 936, 669, 936, 1828, 469, 936, 280, 748, 401, 936, 270, 936, 620, 936, 635, 730, 936, 280, 936, 936, 399, 936, 1325, 839, 936, 933, 936, 876, 936, 1399, 597, 691, 460, 1051, 286, 496, 529, 633, 1115, 936, 489, 91, 936, 1317, 1229, 936, 936, 257, 1391, 726, 936, 629, 504, 205, 514, 907, 322, 936, 744, 854, 140, 324, 936, 1369, 936, 327, 936, 936, 427, 936, 438, 936, 433, 700, 936, 936, 936, 936, 936, 936, 936, 936, 771, 936, 936, 916, 936, 936, 148, 279, 936, 936, 116, 936, 436, 943, 936, 936, 1235, 751, 239, 33, 1385, 936, 901, 936, 308, 443, 936, 936, 297, 808, 804, 112, 482, 621, 107, 1046, 87, 504, 1234, 936, 577, 547, 464, 936, 108, 936, 936, 936, 278, 936, 936, 936, 373, 936, 868, 936, 852, 936, 408, 400, 369, 439, 402, 191, 451, 384, 437, 411, 378, 431, 438, 371, 334, 836, 396, 440, 340, 544, 317, 413, 392, 443, 426, 445, 369, 318, 286, 429, 450, 834, 375]
  class_weights = []
  for count in b:
      weight = 1 / (count / sum(b))
      class_weights.append(weight)
  class_weights = torch.FloatTensor(class_weights).to(device)
  criterion = FocalLoss(alpha=class_weights, gamma=2).to(device)
else:criterion = nn.CrossEntropyLoss().to(device)
if flag_relay:criterion = check_point['criterion']
val_criterion=nn.CrossEntropyLoss().to(device)
print(f"device={device}")
print(torch.cuda.get_device_name(0))
with open(f"/home/yif22003/food_imaging/nutrient/{log_str}.txt","a") as ff:
  ff.write(f"model={model_str},device={device},device name={torch.cuda.get_device_name(0)}\n")
with open("/home/yif22003/food_imaging/nutrient/new nutrient_label.txt","r") as f:
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
if flag_relay:finished_num_epochs=check_point['epoch']+1
else:finished_num_epochs=0
if flag_relay:best_val_loss=check_point['best_val_loss']
else:best_val_loss=99999
if flag_relay:best_val_acc=check_point['best_val_acc']
else:best_val_acc=0
# writer = SummaryWriter()
start_time=time.time()
for epoch in range(finished_num_epochs,finished_num_epochs+num_epochs):
    if epoch==T_max:break
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
        if  qwe%(max(3,int(0.3*n_v2)))==0:
            print(f"epoch={epoch+1},training iter={qwe}/{n_v2}={round(100*qwe/n_v2,2)}%,device={device},elapsed_time={time.time()-start_time}")
            with open(f"/home/yif22003/food_imaging/nutrient/{log_str}.txt","a") as ff:
              ff.write(f"epoch={epoch+1},training iter={qwe}/{n_v2}={round(100*qwe/n_v2,2)}%,device={device},elapsed_time={time.time()-start_time}\n")
        optimizer.zero_grad()
        output = model(x)
        if flag_use_focal_loss:
          loss = criterion(output, y2,y)
        else:loss = criterion(output, y2)
        loss.backward()
        optimizer.step()
        # writer.add_scalar('Memory/Allocated', torch.cuda.memory_allocated(), epoch * n_v2 + step)
        # writer.add_scalar('Memory/Reserved', torch.cuda.memory_reserved(), epoch * n_v2 + step)
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
    num_nutrient_classes=nutrient_label.shape[1]
    r2=np.zeros(num_nutrient_classes,)
    r2_top5=np.zeros(num_nutrient_classes,)
    spearmanr = np.zeros(num_nutrient_classes,)
    pearsonr = np.zeros(num_nutrient_classes,)
    kendalltau = np.zeros(num_nutrient_classes,)
    slope=np.zeros(num_nutrient_classes,)
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
            y=y.cpu()
            batch_nutrient_label=nutrient_label[y]
            batch_output= np.array(torch.nn.functional.softmax(output,dim=1).tolist())
            c=np.argsort(batch_output,axis=1)[:,-5:]
            e=[]
            ty=-1
            for list_idx in c:
                ty+=1
                e.append(batch_output[ty][list_idx])
            e=np.array(e)
            f=e.sum(axis=1)
            g=[]
            ty=-1
            for i in f:
                ty+=1
                g.append(e[ty]/i)
            batch_output_top5=np.array(g)
            batch_nutrient_label_top5=[]
            ty=-1
            for list_idx in c:
                ty+=1
                batch_nutrient_label_top5.append(nutrient_label[list_idx])
            batch_nutrient_label_top5=np.array(batch_nutrient_label_top5)
            batch_nutrient_top5=[]
            ty=-1
            for normalized_top5_prob in batch_output_top5:
                ty+=1
                batch_nutrient_top5.append(np.dot(normalized_top5_prob,batch_nutrient_label_top5[ty]))
            batch_nutrient_top5=np.array(batch_nutrient_top5)
            batch_nutrient=np.dot(batch_output,nutrient_label)
            spearmanr_list=[]
            pearsonr_list=[]
            slope_list=[]
            kendalltau_list=[]
            r2_list=[]
            r2_top5_list=[]
            for nutrient_idx in range(num_nutrient_classes):
                spearmanr_list.append(stats.spearmanr(batch_nutrient[:,nutrient_idx], batch_nutrient_label[:,nutrient_idx]).statistic) # using arithmetic mean nutrient estimation
                pearsonr_corr, _ = stats.pearsonr(batch_nutrient[:,nutrient_idx], batch_nutrient_label[:,nutrient_idx])
                pearsonr_list.append(pearsonr_corr)
                kendalltau_corr, _ = stats.kendalltau(batch_nutrient[:,nutrient_idx], batch_nutrient_label[:,nutrient_idx])
                kendalltau_list.append(kendalltau_corr)
                ransac.fit(batch_nutrient[:,nutrient_idx].reshape(-1, 1), batch_nutrient_label[:,nutrient_idx].reshape(-1, 1))
                slope_ = ransac.estimator_.coef_
                slope_list.append(float(slope_))
                r2_list.append(r2_score(batch_nutrient_label[:,nutrient_idx],batch_nutrient[:,nutrient_idx]))
                r2_top5_list.append(r2_score(batch_nutrient_label[:,nutrient_idx],batch_nutrient_top5[:,nutrient_idx]))
            spearmanr+=np.array(spearmanr_list)
            pearsonr+=np.array(pearsonr_list)
            kendalltau+=np.array(kendalltau_list)
            slope+=np.array(slope_list)
            r2+=np.array(r2_list)
            r2_top5+=np.array(r2_top5_list)
            across_nutrients_mean_spearmanr=np.mean(spearmanr)
            across_nutrients_mean_pearsonr=np.mean(pearsonr)
            across_nutrients_mean_kendalltau=np.mean(kendalltau)
            across_nutrients_mean_slope=np.mean(slope)
            across_nutrients_mean_r2=np.mean(r2)
            across_nutrients_mean_r2_top5=np.mean(r2_top5)
            _,idx_top5=torch.topk(torch.nn.functional.softmax(output,dim=1),5,1)
            batch_val_acc_top5=0
            ty=-1
            for gt in y:
                ty+=1
                if gt in idx_top5[ty]:batch_val_acc_top5+=1
            batch_val_acc_top5/=len(y)
            val_acc_top5+=batch_val_acc_top5
            if rerer%(max(1,int(0.4*n_v)))==0:
                print(f"epoch={epoch+1},validating iter={rerer}/{n_v}={round(100*rerer/n_v,2)}%,device={device},elapsed_time={time.time()-start_time}")
                with open(f"/home/yif22003/food_imaging/nutrient/{log_str}.txt","a") as ff:
                  ff.write(f"epoch={epoch+1},validating iter={rerer}/{n_v}={round(100*rerer/n_v,2)}%,device={device},elapsed_time={time.time()-start_time},val_acc_top5={val_acc_top5/rerer},mean_slope={across_nutrients_mean_slope/rerer},mean_r2={across_nutrients_mean_r2/rerer},mean_r2_top5={across_nutrients_mean_r2_top5/rerer},mean_kendalltau={across_nutrients_mean_kendalltau/rerer},mean_spearmanr={across_nutrients_mean_spearmanr/rerer},mean_pearsonr={across_nutrients_mean_pearsonr/rerer}\n")
    r2/=n_v
    r2_top5/=n_v
    spearmanr/= n_v
    pearsonr /= n_v
    kendalltau/= n_v
    slope  /= n_v   
    val_loss /= n_v
    val_accuracy /= n_v
    val_acc_top5/=n_v
    scheduler.step()
    print(f"End of epoch{epoch + 1},val_accuracy={val_accuracy},train_accuracy={train_acc},train_loss={v1},val_loss:{val_loss},lr={optimizer.param_groups[0]['lr']},device={device},elapsed_time={time.time()-start_time}")
    with open(f"/home/yif22003/food_imaging/nutrient/{log_str}.txt","a") as ff:
      ff.write(f"End of epoch{epoch + 1},val_accuracy={val_accuracy},train_accuracy={train_acc},train_loss={v1},val_loss:{val_loss},lr={optimizer.param_groups[0]['lr']},device={device},elapsed_time={time.time()-start_time},r2_top5={np.mean(r2_top5)},val_acc_top5={val_acc_top5},mean_kendalltau={np.mean(kendalltau)},mean_spearmanr={np.mean(spearmanr)},mean_pearsonr={np.mean(pearsonr)},r2={np.mean(r2)},mean_slope={np.mean(slope)},train_acc_top5={train_acc_top5}\n")
      ff.write(f"End of epoch{epoch + 1},r2={r2}\n")
      ff.write(f"End of epoch{epoch + 1},r2_top5={r2_top5}\n")
      ff.write(f"End of epoch{epoch + 1},slope={slope}\n")
      ff.write(f"End of epoch{epoch + 1},spearmanr={spearmanr}\n")
      ff.write(f"End of epoch{epoch + 1},pearsonr={pearsonr}\n")
      ff.write(f"End of epoch{epoch + 1},kendalltau={kendalltau}\n")
    if val_accuracy>=best_val_acc or val_loss<=best_val_loss:
      best_val_acc=val_accuracy
      best_val_loss=val_loss
      state_path=f"/home/yif22003/food_imaging/nutrient/{log_str}_state_best.pt"
      state = {
      'epoch': epoch,
      'model_dict': model.state_dict(),
      'optimizer_dict': optimizer.state_dict(),
      'scheduler_dict': scheduler.state_dict(),
      'criterion': criterion,
      'best_val_loss': best_val_loss,
      'best_val_acc': best_val_acc
      }
      torch.save(state,state_path)
      with open(f"/home/yif22003/food_imaging/nutrient/{log_str}.txt","a") as ff:
        ff.write(f"{log_str} saved after epoch{epoch+1} with val_acc = {best_val_acc} and val_loss={best_val_loss}\n")
    else:
      state_path=f"/home/yif22003/food_imaging/nutrient/{log_str}_state_relay.pt"
      state = {
      'epoch': epoch,
      'model_dict': model.state_dict(),
      'optimizer_dict': optimizer.state_dict(),
      'scheduler_dict': scheduler.state_dict(),
      'criterion': criterion,
      'best_val_loss': best_val_loss,
      'best_val_acc': best_val_acc
      }
      torch.save(state,state_path)
      with open(f"/home/yif22003/food_imaging/nutrient/{log_str}.txt","a") as ff:
        ff.write(f"{log_str} saved after epoch{epoch+1} with other model best_val_acc = {best_val_acc} and best_val_loss={best_val_loss}\n")
# writer.close()