import torch
import time
import numpy as np

#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model
from GreedyInfoMax.audio.data import get_dataloader
from GreedyInfoMax.audio.validation import val_by_latent_speakers
from GreedyInfoMax.audio.validation import val_by_CPC, val_by_SVM


def train(opt, model):
    total_step = len(train_loader)

    # how often to output training values
    print_idx = 100
    # how often to validate training process by plotting latent representations of various speakers
    latent_val_idx = 1000

    starttime = time.time()

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for i in range(opt.model_splits)]

        # validate training progress by training and testing an SVM on a subset of the speakers
        if opt.SVM_training_samples > 0:
            speaker_accuracy = val_by_SVM.val_by_SVM_speaker_classification(
                opt, model, train_dataset, test_dataset
            )
            logs.append_SVM_acc(speaker_accuracy)

        for step, (audio, filename, _, start_idx) in enumerate(train_loader):

            # validate training progress by plotting latent representation of various speakers
            if step % latent_val_idx == 0:
                val_by_latent_speakers.val_by_latent_speakers(
                    opt, train_dataset, model, epoch, step
                )

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = audio.to(opt.device)

            loss = model(model_input, filename, start_idx, n=opt.train_layer)
            loss = torch.mean(loss, 0)  # average over the losses from different GPUs

            for idx, cur_losses in enumerate(loss):
                model.zero_grad()

                if idx == len(loss) - 1:
                    cur_losses.backward()
                else:
                    cur_losses.backward(retain_graph=True)
                optimizer[idx].step()

                print_loss = cur_losses.item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))

                loss_epoch[idx] += print_loss

        logs.append_train_loss([x / total_step for x in loss_epoch])

        # validate by testing the CPC performance on the validation set
        if opt.validate:
            validation_loss = val_by_CPC.val_by_CPC(opt, model, test_loader)
            logs.append_val_loss(validation_loss)

        logs.create_log(model, epoch=epoch, optimizer=optimizer)


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)

    # set random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load model
    model, optimizer = load_audio_model.load_model_and_optimizer(opt)

    # initialize logger
    logs = logger.Logger(opt)

    # get datasets and dataloaders
    train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_libri_dataloaders(
        opt
    )

    try:
        # Train the model
        train(opt, model)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)
