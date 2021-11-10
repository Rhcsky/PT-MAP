import torch
from tqdm import tqdm

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas


class PowerTransform:
    def centerDatas(self, data):
        data[:n_lsamples] -= data[:n_lsamples].mean(0, keepdim=True)
        data[:n_lsamples] /= torch.norm(data[:n_lsamples], 2, 1).unsqueeze(1)
        data[n_lsamples:] -= data[n_lsamples:].mean(0, keepdim=True)
        data[n_lsamples:] /= torch.norm(data[n_lsamples:], 2, 1).unsqueeze(1)

        # data -= data.mean(0, keepdim=True)
        # data /= torch.norm(data, 2, 1).unsqueeze(1)
        return data

    def scaleEachUnitaryDatas(self, data):
        norms = data.norm(dim=1, keepdim=True)
        return data / norms

    def QRreduction(self, data):
        ndata = torch.linalg.qr(data.permute(1, 0)).R
        ndata = ndata.permute(1, 0)
        return ndata

    def controlUnderflow(self, data, epsilon=1e-6, beta=0.5):
        data = torch.pow(data + epsilon, beta)
        return data

    def __call__(self, data):
        data = self.controlUnderflow(data)
        # data = self.QRreduction(data)
        data = self.scaleEachUnitaryDatas(data)
        data = self.centerDatas(data)
        return data


def optimal_transport(prob):
    P = torch.exp(-10 * prob)
    P /= P.view(-1, 1).sum(0).unsqueeze(1)
    return P


if __name__ == '__main__':
    # ---- data loading
    n_shot = 5
    n_ways = 5
    n_queries = 15
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet("miniimagenet")
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    print(f'NDATAS: {ndatas.shape}')
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    print(f'NDATAS: {ndatas.shape}')
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)

    # switch to cuda
    ndatas = ndatas.to('cuda:6')
    labels = labels.to('cuda:6')

    acc_list = list()
    pt = PowerTransform()

    for data, label in tqdm(zip(ndatas, labels)):
        # data = pt(data)
        sup = data.reshape(20, 5, -1)[:5].transpose(0, 1).mean(1)
        qry = data[n_lsamples:]
        label = label[n_lsamples:]

        dist = (qry.unsqueeze(1) - sup.unsqueeze(0)).norm(dim=2).pow(2)
        prob = optimal_transport(dist)

        y_pred = prob.argmax(1)
        accuracy = y_pred.eq(label).float().mean()
        acc_list.append(accuracy.item())

    import numpy as np

    print(np.array(acc_list).mean())
