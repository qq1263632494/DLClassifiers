import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from ProgressBar import ShowProcess
from sklearn.metrics import accuracy_score


class Classifier:
    def __init__(self, nn=None):
        self.nn = nn.cuda()

    def fit(self, train_set, batch_size, optim, loss_func, epoch, lr):
        global optimizer
        if optim == 'adam':
            optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
        if optim == 'sgd':
            optimizer = torch.optim.SGD(self.nn.parameters(), lr=lr)
        if optim == 'asgd':
            optimizer = torch.optim.ASGD(self.nn.parameters(), lr=lr)
        if optim == 'adadelta':
            optimizer = torch.optim.Adadelta(self.nn.parameters(), lr=lr)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        i = 0
        list_x = []
        list_y = []
        bar = ShowProcess(epoch * len(train_set) / batch_size)
        for t in range(epoch):
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x_t = b_x.cuda()
                b_y_t = b_y.cuda()
                # b_x_t = b_x
                # b_y_t = b_y
                output = self.nn(b_x_t)
                loss = loss_func(output, b_y_t)
                list_x.append(i)
                list_y.append(loss.item())
                i += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                msg = '，当前误差：%.5f' % loss.data.cpu().numpy()
                bar.show_process(msg=msg)
        bar.close()
        print('\033[1;32m初始误差：\033[0m' + '\033[1;31m%.5f\033[0m' % list_y[0])
        print('\033[1;32m最终误差：\033[0m' + '\033[1;31m%.5f\033[0m' % list_y[-1])
        msg = str(type(self.nn))
        msg += '\n solver=' + optim + \
               ' epoch=' + str(epoch) + \
               ' learning rate =' + str(lr) + ' batch=' + str(batch_size)
        plt.title(msg)
        plt.plot(list_x[10:], list_y[10:])
        plt.savefig('pic.png', bbox_inches='tight')
        plt.show()

    def load(self, path):
        print('加载模型：' + path)
        self.nn = torch.load(path)
        print('加载完毕')

    def save(self, path):
        torch.save(self.nn, path)

    def evaluate(self, test_set, batch_size_test):
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        list_pred = []
        list_true = []
        for step, (t_x, t_y) in enumerate(test_loader):
            print(t_x, t_y)
            t_x_t = t_x.cuda()
            test_output = self.nn(t_x_t)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
            list_pred.append(pred_y)
            list_true.append(t_y)
        print(accuracy_score(y_true=list_true, y_pred=list_pred))

    def evaluate2(self, X_test, Y_test, batch_size_test):
        length = len(X_test)
        list_pred = []
        iter_times = length / batch_size_test
        iter_times = int(iter_times)
        i = 0
        while i < iter_times:
            t_x = torch.FloatTensor(X_test[i * batch_size_test:(i + 1) * batch_size_test])
            t_x_t = t_x.cuda()
            test_output = self.nn(t_x_t)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
            list_pred += pred_y.tolist()
            i += 1
        t_x = torch.FloatTensor(X_test[i * batch_size_test:])
        t_x_t = t_x.cuda()
        test_output = self.nn(t_x_t)
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        list_pred += pred_y.tolist()
        print(accuracy_score(y_true=Y_test, y_pred=list_pred))

    def fit_with_LBFGS(self, train_set, batch_size, loss_func, epoch, lr):
        optimizer = torch.optim.LBFGS(self.nn.parameters(), lr=lr)
        loss_func = torch.nn.CrossEntropyLoss()
        list_x = []
        list_y = []
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        from ProgressBar import ShowProcess
        bar = ShowProcess(epoch * len(train_set) / batch_size)
        for t in range(epoch):
            for step, (b_x, b_y) in enumerate(train_loader):
                def closure():
                    b_x_t = b_x.cuda()
                    b_y_t = b_y.cuda()
                    output = self.nn(b_x_t)
                    loss = loss_func(output, b_y_t)
                    list_y.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    return loss

                optimizer.step(closure)
                bar.show_process()
        bar.close()
        for i in range(len(list_y)):
            list_x.append(i)
        print(list_y[0])
        print(list_y[-1])
        msg = str(type(self.nn))
        msg += '\n solver= lbgfs' + ' epoch=' + str(epoch) + ' learning rate =' + str(lr)
        plt.title(msg)
        plt.plot(list_x, list_y)
        plt.savefig('pic.png', bbox_inches='tight')
        plt.show()

    def predict(self, features, batch_size_test):
        length = len(features)
        list_pred = []
        iter_times = length / batch_size_test
        iter_times = int(iter_times)
        for i in range(iter_times):
            t_x = torch.FloatTensor(features[i * batch_size_test:(i + 1) * batch_size_test])
            t_x_t = t_x.cuda()
            test_output = self.nn(t_x_t)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
            list_pred += pred_y.tolist()
        return list_pred

    def save_model_dicts(self, path):
        torch.save(self.nn.state_dict(), path)

    def load_model_dicts(self, path, c):
        self.nn = c
        self.nn.load_state_dict(torch.load(path))
        self.nn.eval()
        self.nn.cuda()