from data import KidsDataset
import model
import utils
import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchsummary import summary

model_path = r'./final_model/'
data_path = r'./dataset_test//'
dataset = KidsDataset(path=data_path, window_size=600, is2D=True, is4divide=False)

testLoader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
net = model.DenseNet_121().cuda()

_, _, _, epoch_str, global_step = utils.load_checkpoint(utils.latest_checkpoint_path(model_path, "epoch_*.pth"), net)
net.eval()
summary(net, (1, 12, 150), device='cuda')

true_labels = []
pred_labels = []

total = 0
correct = 0

for i, data in enumerate(tqdm.tqdm(testLoader)):
    with torch.no_grad():
        output = net(data['data'].cuda()).cpu()
        _, predicted = torch.max(output, 1)
        true_labels.append(data['label'].numpy())
        pred_labels.append(predicted.numpy())
        total += data['label'].size(0)
        correct += (predicted == data['label']).sum()

accuracy = 100 * correct / total
cm = confusion_matrix(true_labels, pred_labels, labels=range(0,13), normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1,14))
disp.plot(cmap=plt.cm.Blues)
plt.title("accuracy = " + str(accuracy.numpy()) + "%")
plt.show()