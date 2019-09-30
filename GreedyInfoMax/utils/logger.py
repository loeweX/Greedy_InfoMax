import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy


class Logger:
    """
    Logging class that keeps track of the training parameters and progress, and saves log files and model checkpoints
    """
    def __init__(self, opt):
        self.opt = opt

        if opt.validate:
            self.val_loss = [[] for i in range(opt.model_splits)]
        else:
            self.val_loss = None

        if opt.start_epoch > 0:
            self.loss_last_training = np.load(
                os.path.join(opt.model_path, "train_loss.npy")
            ).tolist()
            self.train_loss = copy.deepcopy(self.loss_last_training)
            try:
                self.SVM_accuracy = np.load(
                    os.path.join(opt.model_path, "svm_accuracy.npy")
                ).tolist()
            except:
                self.SVM_accuracy = []
        else:
            self.loss_last_training = None
            self.train_loss = [[] for i in range(opt.model_splits)]
            self.SVM_accuracy = []

        self.num_models_to_keep = 2
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

    def create_log(
        self,
        model,
        accuracy=None,
        epoch=0,
        optimizer=None,
        final_test=False,
        final_loss=None,
        acc5=None,
        predict_model=None
    ):

        print("Saving model and log-file to " + self.opt.log_path)

        # Save the model checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(self.opt.log_path, "model_{}.ckpt".format(epoch)),
        )

        # remove old model files to keep dir uncluttered
        try:
            os.remove(
                os.path.join(
                    self.opt.log_path,
                    "model_{}.ckpt".format(epoch - self.num_models_to_keep),
                )
            )
        except:
            print("not enough models there yet, nothing to delete")

        if predict_model is not None:
            # Save the predict model checkpoint
            torch.save(
                predict_model.state_dict(),
                os.path.join(self.opt.log_path, "predict_model_{}.ckpt".format(epoch)),
            )

            # remove old model files to keep dir uncluttered
            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "predict_model_{}.ckpt".format(epoch - self.num_models_to_keep),
                    )
                )
            except:
                print("not enough models there yet, nothing to delete")

        if optimizer is not None:
            for idx, optims in enumerate(optimizer):
                torch.save(
                    optims.state_dict(),
                    os.path.join(
                        self.opt.log_path, "optim_{}_{}.ckpt".format(idx, epoch)
                    ),
                )

                try:
                    os.remove(
                        os.path.join(
                            self.opt.log_path,
                            "optim_{}_{}.ckpt".format(
                                idx, epoch - self.num_models_to_keep
                            ),
                        )
                    )
                except:
                    print("not enough models there yet, nothing to delete")

        # Save hyper-parameters
        with open(os.path.join(self.opt.log_path, "log.txt"), "w+") as cur_file:
            cur_file.write(str(self.opt))
            if accuracy is not None:
                cur_file.write("Top 1 - accuracy: " + str(accuracy))
            if acc5 is not None:
                cur_file.write("Top 5 - Accuracy: " + str(acc5))
            if final_test and accuracy is not None:
                cur_file.write(" Very Final testing accuracy: " + str(accuracy))
            if final_test and acc5 is not None:
                cur_file.write(" Very Final testing top 5 - accuracy: " + str(acc5))

        # Save losses throughout training and plot
        np.save(
            os.path.join(self.opt.log_path, "train_loss"), np.array(self.train_loss)
        )
        np.save(
            os.path.join(self.opt.log_path, "svm_accuracy"),
            np.array(self.SVM_accuracy),
        )

        if self.val_loss is not None:
            np.save(
                os.path.join(self.opt.log_path, "val_loss"), np.array(self.val_loss)
            )

        self.draw_loss_curve()
        self.draw_svm_accuracies_curve(self.SVM_accuracy, "accuracy")

        if accuracy is not None:
            np.save(os.path.join(self.opt.log_path, "accuracy"), accuracy)

        if final_test:
            np.save(os.path.join(self.opt.log_path, "final_accuracy"), accuracy)
            np.save(os.path.join(self.opt.log_path, "final_loss"), final_loss)

    def draw_svm_accuracies_curve(self, svm_accuracies, label):
        lst_iter = np.arange(len(svm_accuracies))
        plt.plot(lst_iter, np.array(svm_accuracies), "-b", label="SVM " + label)

        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="upper right")

        # save image
        plt.savefig(os.path.join(self.opt.log_path, "svm_accuracy_" + label + ".png"))
        plt.close()

    def draw_loss_curve(self):
        for idx, loss in enumerate(self.train_loss):
            lst_iter = np.arange(len(loss))
            plt.plot(lst_iter, np.array(loss), "-b", label="train loss")

            if self.loss_last_training is not None:
                lst_iter = np.arange(len(self.loss_last_training[idx]))
                plt.plot(lst_iter, self.loss_last_training[idx], "-g")

            if self.val_loss is not None:
                lst_iter = np.arange(len(self.val_loss[idx]))
                plt.plot(lst_iter, np.array(self.val_loss[idx]), "-r", label="val loss")

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc="upper right")

            # save image
            plt.savefig(os.path.join(self.opt.log_path, "loss_{}.png".format(idx)))
            plt.close()

    def append_SVM_acc(self, SVM_acc):
        self.SVM_accuracy.append(SVM_acc)

    def append_train_loss(self, train_loss):
        for idx, elem in enumerate(train_loss):
            self.train_loss[idx].append(elem)

    def append_val_loss(self, val_loss):
        for idx, elem in enumerate(val_loss):
            self.val_loss[idx].append(elem)
