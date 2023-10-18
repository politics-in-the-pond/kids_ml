import os
import torch
import gc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class KidsDataset(Dataset):
    x_data = []
    y_data = []
    correspondence = {1:1, 2:2, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:3, 11:4, 12:2, 13:1}

    def __init__(self, path, window_size=50, is2D=False, is4divide=False):
        super(KidsDataset, self).__init__()
        self.window_size = window_size
        dir_path = os.listdir(path)
        #dir_path.remove('2')
        #dir_path.remove('5')
        for d in dir_path:
            file_path = os.listdir(path + '/' + d)
            for f in file_path:
                lines = []
                with open(path + '/' + d + '/' + f, 'r') as file:
                    for line in file:
                        lines.append(line)

                df = {}
                for l in lines:
                    tmp = l.split(",")
                    name = tmp[0]
                    tmp = tmp[1:]
                    nums = []
                    for t in tmp:
                        nums.append(float(t))
                    df[name] = nums
                for value in range(0, len(df['filteredAccelX']) - window_size + 1):
                    names = ['rawAccelX', 'rawAccelY', 'rawAccelZ', 'rawGyroX', 'rawGyroY', 'rawGyroZ', 'scaledAccelX', 'scaledAccelY', 'scaledAccelZ', 'scaledGyroX', 'scaledGyroY', 'scaledGyroZ', 'filteredAccelX', 'filteredAccelY', 'filteredAccelZ', 'filteredGyroX', 'filteredGyroY', 'filteredGyroZ']
                    datas = []

                    for n in names:
                        datas.append(torch.tensor(df[n][value:value + window_size]))

                    try:
                        if is2D:
                            self.x_data.append(torch.unsqueeze(torch.stack(datas, 0), 0))
                        else:
                            self.x_data.append(torch.transpose(torch.stack(datas, 0), 0, 1))
                        if not is4divide:
                            self.y_data.append(int(d)-1)
                        else:
                            self.y_data.append(self.correspondence[int(d)]-1)
                    except:
                        pass
                gc.collect()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, item):
        return {'data' : self.x_data[item], 'label' : self.y_data[item]}