# import  torchvision.transforms as transforms
# from    PIL import Image
# import  os.path
import  numpy as np
# import  scipy.io as sio
import  time


class deepMIMONshot:

    def __init__(self, args, x_train, y_train, x_test, y_test):


        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batchsz = args.task_num
        self.n_way = args.n_way     # n way
        self.k_shot = args.k_spt    # k shot
        self.k_query = args.k_qry   # k query
        self.n_antenna = args.n_antenna # n antennas
        assert (self.k_shot + self.k_query) <= 100  # or else exceed the feasible range

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": [x_train, y_train],
                         "test":  [x_test, y_test]}
        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),
                                "test": self.load_data_cache(self.datasets["test"])}

    def load_data_cache(self, data_pack):
        
        spt_size = self.k_shot * self.n_way
        qry_size = self.k_query * self.n_way
        data_cache = []
        data = data_pack[0]     # training or test data
        label = data_pack[1]    # training or test label
        n_row = data.shape[0]
        n_ue = data.shape[1]
        for _ in range(10):      # number of storage samples in cache
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_class = np.random.choice(n_row, self.n_way, False)

                for _, current_class in enumerate(selected_class):
                    selected_grid = np.random.choice(n_ue, self.k_shot + self.k_query, False)

                    x_spt.append(data[current_class][selected_grid[:self.k_shot]])   # (1, 1, 64) -> (5, 1, 64)
                    x_qry.append(data[current_class][selected_grid[self.k_shot:]])
                    y_spt.append(label[current_class][selected_grid[:self.k_shot]])
                    y_qry.append(label[current_class][selected_grid[self.k_shot:]])
                
                x_spt = np.array(x_spt).reshape(spt_size, self.n_antenna * 2)    # 1-5
                y_spt = np.array(y_spt).reshape(spt_size)
                x_qry = np.array(x_qry).reshape(qry_size, self.n_antenna * 2)    # 1-15, 16-30, ..., 61-75
                y_qry = np.array(y_qry).reshape(qry_size)

                # append [spt_size, 128] -> [batchsize, spt_size, 128]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, spt_size, self.n_antenna * 2)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, spt_size)
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, qry_size, self.n_antenna * 2)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, qry_size)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache


    def next(self, mode='train'):

        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

if __name__ == '__main__':

    time.sleep(10)


