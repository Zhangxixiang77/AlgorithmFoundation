import pandas as pd
import numpy as np
import json
import csv
import os
from tqdm import tqdm
import csv


# 评分预测    1-5
class MatrixDecomForRecSys(object):

    def __init__(
        self,
        lr,
        batch_size,
        reg_p,
        reg_q,
        reg_b,
        gamma,
        hidden_size=10,
        epoch=10,
        columns=["uid", "iid", "rating"],
        metric=None,
        bias=False,
    ):
        self.lr = lr  # 学习率
        self.batch_size = batch_size
        self.reg_p = reg_p  # P矩阵正则系数
        self.reg_q = reg_q  # Q矩阵正则系数
        self.reg_b = reg_b  # 偏置项正则系数
        self.gamma = gamma  # 协同过滤系数
        self.hidden_size = hidden_size  # 隐向量维度
        self.epoch = epoch  # 最大迭代次数
        self.columns = columns
        self.metric = metric
        self.bias = bias

    def load_dataset(self, train_data, dev_data):
        self.train_data = pd.DataFrame(train_data)
        self.dev_data = pd.DataFrame(dev_data)

        self.users_ratings = train_data.groupby(self.columns[0]).agg(
            [list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = train_data.groupby(self.columns[1]).agg(
            [list])[[self.columns[0], self.columns[2]]]

        R = pd.DataFrame(
            0.,
            index=self.users_ratings.index,
            columns=self.items_ratings.index,
        )
        for j in tqdm(self.items_ratings.index, desc='Constructing R'):
            R.loc[self.items_ratings.loc[j, ('userId', 'list')],
                  j] = self.items_ratings.loc[j, ('rating', 'list')]
        self.R = R.values
        self.globalMean = self.train_data[self.columns[2]].mean()

    def _init_matrix(self):
        P_ = np.random.rand(
            len(self.users_ratings),
            self.hidden_size,
        )
        Q_ = np.random.rand(
            len(self.items_ratings),
            self.hidden_size,
        )
        B_ = np.zeros((
            len(self.users_ratings),
            len(self.items_ratings),
        ))
        return P_, Q_, B_

    def train(self, optimizer_type: str):
        P_, Q_, B_ = self._init_matrix()  # 初始化user、item矩阵
        best_metric_result = None
        best_P, best_Q, best_B = (
            dict(zip(self.users_ratings.index, P_)),
            dict(zip(self.items_ratings.index, Q_)),
            pd.DataFrame(
                B_,
                index=self.users_ratings.index,
                columns=self.items_ratings.index,
            ),
        )

        for i in range(self.epoch):
            print("Epoch: %d" % i)
            # 当前epoch，执行优化算法：
            if optimizer_type == "SGD":  # 随机梯度下降
                P_, Q_, B_ = self.sgd(P_, Q_, B_)
            elif optimizer_type == "BGD":  # 批量梯度下降
                P_, Q_, B_ = self.bgd(P_, Q_, B_, batch_size=self.batch_size)
            else:
                raise NotImplementedError("Please choose one of SGD and BGD.")

            # 当前epoch优化后，在验证集上验证，并保存目前最好的P和Q
            P = dict(zip(self.users_ratings.index, P_))
            Q = dict(zip(self.items_ratings.index, Q_))
            B = pd.DataFrame(
                B_,
                index=self.users_ratings.index,
                columns=self.items_ratings.index,
            )
            metric_result = self.eval(P, Q, B)
            # 如果当前的RMSE更低，则保存
            print("Current dev metric result: {}".format(metric_result))
            if best_metric_result is None or metric_result <= best_metric_result:
                best_metric_result = metric_result
                best_P, best_Q, best_B = P, Q, B
                print("Best dev metric result: {}".format(best_metric_result))
            

        # 最后保存最好的P和Q
        np.savez("best_pq.npz", P=best_P, Q=best_Q, B=best_B)

    def sgd(self, P_, Q_, B_):
        
        residual = P_ @ Q_.T + B_ - self.R
        grad_p = residual @ Q_
        grad_q = residual.T @ P_
        # 偏置项
        grad_b = residual

        # 正则化
        grad_p += self.reg_p * P_
        grad_q += self.reg_q * Q_
        grad_b += self.reg_b * B_

        # 协同过滤
        if self.gamma > 0.:
            G = P_ @ P_.T
            Gii = np.diagonal(G)
            D = np.c_[Gii] + Gii - G
            G_div_D2 = np.square(G / D)
            grad_filter = np.c_[G_div_D2.sum(
                1) - 2 / np.sum(1 / D, 1)] * P_ - G_div_D2 @ P_ + P_.sum(0)
            grad_p += self.gamma * grad_filter

        P_ -= self.lr * grad_p
        Q_ -= self.lr * grad_q
        if self.bias:
            B_ -= self.lr * grad_b

        return P_, Q_, B_

    def bgd(self, P_, Q_, B_, batch_size: int = 8):
        
        for i in tqdm(range(0, P_.shape[0], batch_size), desc="BGD"):
            batch_P = P_[i:i + batch_size]
            batch_B = B_[i:i + batch_size]
            batch_R = self.R[i:i + batch_size]
            residual = batch_P @ Q_.T + batch_B - batch_R

            grad_p = residual @ Q_
            grad_q = residual.T @ batch_P
            grad_b = residual

            grad_p += self.reg_p * batch_P
            grad_q += self.reg_q * Q_
            grad_b += self.reg_b * batch_B

            if self.gamma > 0.:
                G = batch_P @ batch_P.T
                Gii = np.diagonal(G)
                D = np.c_[Gii] + Gii - G
                G_div_D2 = np.square(G / D)
                grad_filter = np.c_[G_div_D2.sum(1) - 2 / np.sum(
                    1 / D, 1)] * batch_P - G_div_D2 @ batch_P + batch_P.sum(0)
                grad_p += self.gamma * grad_filter

            P_[i:i + batch_size] -= self.lr * grad_p
            Q_ -= self.lr * grad_q
            if self.bias:
                B_[i:i + batch_size] -= self.lr * grad_b
        return P_, Q_, B_

    def predict_user_item_rating(self, uid, iid, P, Q, B):
        # 如果uid或iid不在，我们使用全局平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = P[uid]
        q_i = Q[iid]

        return np.dot(p_u, q_i) + B.loc[uid, iid]

    def eval(self, P, Q, B):
        # 根据当前的P和Q，在dev上进行验证，挑选最好的P和Q向量
        dev_loss = 0.
        prediction, ground_truth = list(), list()
        for uid, iid, real_rating in self.dev_data.itertuples(index=False):
            prediction_rating = self.predict_user_item_rating(
                uid, iid, P, Q, B)
            # dev_loss += abs(prediction_rating - real_rating)
            prediction.append(prediction_rating)
            ground_truth.append(real_rating)

        metric_result = self.metric(ground_truth, prediction)

        return metric_result

    def test(self, test_data):
        test_data = pd.DataFrame(test_data)
        # 加载训练好的P和Q
        best_pq = np.load("best_pq.npz", allow_pickle=True)
        P, Q, B = best_pq["P"][()], best_pq["Q"][()], best_pq["B"][()]

        B = pd.DataFrame(B,index=self.users_ratings.index,columns=self.items_ratings.index,)

        save_results = list()
        for uid, iid in test_data.itertuples(index=False):
            pred_rating = self.predict_user_item_rating(uid, iid, P, Q, B)
            save_results.append(pred_rating)

        log_path = "submit_results.csv"
        if os.path.exists(log_path):
            os.remove(log_path)
        file = open(log_path, 'a+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'ID', 'rating'])
        for ei, rating in enumerate(save_results):
            csv_writer.writerow([ei, rating])
        file.close()
