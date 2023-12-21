import torch

import matplotlib.pyplot as plt
import os as os

class Network(object):
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)

        if self.config.use_cuda:
            self.model.cuda()

        # init auxiliary stuff such as log_func
        self._init_aux()
        print(f"Network mode: {config.mode}")

    def _init_aux(self):
        self.log_func = print

        self.log_dir = self.config.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")

        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

    def plot_log(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Visualization of training logs")

        ax1.plot(self.idx_steps, self.train_losses)

        if self.config.use_cuda:
            valid_oas = [acc.cpu().item() for acc in self.valid_oas]
        else:
            valid_oas = self.valid_oas
        ax2.plot(self.idx_steps, valid_oas)
        ax1.set_title("Training loss curve")
        ax2.set_title("Validation accuracy curve")
        plt.tight_layout()
        plt.show()
        plt.close()
        return fig

    def _save(self, pt_file):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "valid_oas": self.valid_oas,
                "idx_steps": self.idx_steps,
                "config": self.config,
            },
            pt_file,
        )

    def _restore(self, pt_file, restore_training=False):
        print(f"restoring {pt_file}")

        load_res = torch.load(pt_file)
        self.model.load_state_dict(load_res["model"])
        self.optimizer.load_state_dict(load_res["optimizer"])

        if restore_training:
            self.train_losses = load_res["train_losses"]
            self.valid_oas = load_res["valid_oas"]
            self.idx_steps = load_res["idx_steps"]

    def train(self, loader_tr, loader_va):
        self.model.train()
        best_va_acc = 0

        for epoch in range(self.config.num_epochs):
            losses = []
            for data in loader_tr:
                if self.config.use_cuda:
                    for i in range(0, len(data)):
                        data[i] = data[i].cuda()

                labels = data[2].squeeze(-1)
                one_hot_labels = torch.nn.functional.one_hot(labels.type(torch.int64), self.config.num_classes)
                # print(data[2].shape)
                pred = self.model(data[0])

                loss = self.model.get_loss(pred, one_hot_labels.type(torch.float64))
                losses += [loss]

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            loss_avg = torch.mean(torch.stack(losses)).item()

            # Save model every epoch.
            self._save(self.checkpts_file)
            acc = self.test(loader_va, mode="valid")
            if acc > best_va_acc:
                best_va_acc = acc
                self._save(self.bestmodel_file)

            self.log_func(
                "Epoch: %3d, loss_avg: %.5f, val OA: %.5f, best val OA: %.5f"
                %(epoch, loss_avg, acc, best_va_acc)
            )

            # Recording training losses and validation performance.
            self.train_losses += [loss_avg]
            self.valid_oas += [acc]
            self.idx_steps += [epoch]

    def test(self, loader_te, mode="test"):
        if mode == "test":
            self._restore(self.bestmodel_file)
        self.model.eval()

        with torch.no_grad():
            accs = []
            num_samples = 0
            for data in loader_te:
                if self.config.use_cuda:
                    for i in range(0, len(data)):
                        data[i] = data[i].cuda()
                batch_size = len(data[2])
                pred = self.model(data[0])
                acc = self.model.get_acc(pred, data[2].squeeze(-1))
                accs += [acc * batch_size]
                # print(accs)
                num_samples += batch_size
            # print(accs)
            # print(num_samples)
            avg_acc = torch.stack(accs).sum() / num_samples

        self.model.train()
        return avg_acc