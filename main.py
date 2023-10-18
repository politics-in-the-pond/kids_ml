from data import KidsDataset
import model
import utils
import tqdm
import torch
import os
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# -----------------**settings**-------------------
lr = 1e-4
beta1 = 0.5
batch_size = 96
epoch = 100
window_size = 600
save_dur = 1 #저장 간격
is4divide = True #행동 4개 분류 여부
model_name = "test"
data_path = r'./dataset//'
# -----------------**settings**-------------------

os.makedirs(r'./models/'+model_name+"/",exist_ok=True)
model_path = r'./models/'+model_name+"/"

epoch_str = 1
global_step = 0

if __name__ == "__main__":
    #데이터 로딩
    dataset = KidsDataset(path=data_path, window_size=window_size, is2D=True, is4divide=is4divide)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(1234))

    trainLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator().manual_seed(1234))
    valLoader = DataLoader(dataset=validation_dataset, batch_size=int(batch_size/8), num_workers=0, shuffle=False)

    net = model.DenseNet_121().cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.99))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999875, last_epoch=epoch_str - 2)

    CEloss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    writer = SummaryWriter(log_dir=model_path)
    writer_eval = SummaryWriter(log_dir=model_path+"eval/")


    #체크포인트 로딩
    try:
        _, _, _, epoch_str, global_step = utils.load_checkpoint(utils.latest_checkpoint_path(model_path, "epoch_*.pth"), net, optimizer)
        print("loaded model step", global_step, ", epoch :", epoch_str)
    except:
        epoch_str = 1
        global_step = 0

    # 학습
    for e in range(epoch_str, epoch + 1):
        print("epoch " + str(epoch_str) + "==>")
        net.train()

        losses = 0
        total = 0
        correct = 0

        for i, data in enumerate(tqdm.tqdm(trainLoader)):
            global_step += batch_size
            optimizer.zero_grad()
            output = net(data['data'].cuda())
            loss = CEloss(output, data['label'].cuda())
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            losses += loss
            total += data['label'].size(0)
            correct += (predicted == data['label'].cuda()).sum()

        accuracy = 100 * correct / total
        loss = losses / total
        writer.add_scalar("train_loss", loss, global_step)
        writer.add_scalar("train_acc", accuracy, global_step)

        losses = 0
        total = 0
        correct = 0
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(tqdm.tqdm(valLoader)):
                output = net(data['data'].cuda())
                loss = CEloss(output, data['label'].cuda())

                _, predicted = torch.max(output, 1)
                losses += loss
                total += data['label'].size(0)
                correct += (predicted == data['label'].cuda()).sum()

        accuracy = 100 * correct / total
        loss = losses / total
        writer.add_scalar("val_loss", loss, global_step)
        writer.add_scalar("val_acc", accuracy, global_step)

        utils.save_checkpoint(net, optimizer, lr, epoch_str, global_step, os.path.join(model_path, "epoch_{}.pth".format(epoch_str)))
        epoch_str = epoch_str + 1
