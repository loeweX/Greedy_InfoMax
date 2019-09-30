import torch
import time
import os
import numpy as np

## own modules
from GreedyInfoMax.audio.data import get_dataloader, phone_dict
from GreedyInfoMax.utils import logger, utils
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def train(opt, phone_dict, context_model, model):
    total_step = len(train_dataset.file_list)

    for epoch in range(opt.num_epochs):
        loss_epoch = 0

        for i, k in enumerate(train_dataset.file_list):
            starttime = time.time()

            audio, filename = train_dataset.get_full_size_test_item(i)

            ### get latent representations for current audio
            model_input = audio.to(opt.device)
            model_input = torch.unsqueeze(model_input, 0)

            targets = torch.LongTensor(phone_dict[filename])
            targets = targets.to(opt.device).reshape(-1)

            if opt.model_type == 2:  ##fully supervised training
                for idx, layer in enumerate(context_model.module.fullmodel):
                    context, z = layer.get_latents(model_input)
                    model_input = z.permute(0, 2, 1)
            else:
                with torch.no_grad():
                    for idx, layer in enumerate(context_model.module.fullmodel):
                        if idx + 1 < len(context_model.module.fullmodel):
                            _, z = layer.get_latents(
                                model_input, calc_autoregressive=False
                            )
                            model_input = z.permute(0, 2, 1)
                    context, _ = context_model.module.fullmodel[idx].get_latents(
                        model_input, calc_autoregressive=True
                    )
                context = context.detach()

            inputs = context.reshape(-1, n_features)

            # forward pass
            output = model(inputs)

            """ 
            The provided phone labels are slightly shorter than expected, 
            so we cut our predictions to the right length.
            Cutting from the front gave better results empirically.
            """
            output = output[-targets.size(0) :]  # output[ :targets.size(0)]

            loss = criterion(output, targets)

            # calculate accuracy
            accuracy, = utils.accuracy(output.data, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sample_loss = loss.item()
            loss_epoch += sample_loss

            if i % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Accuracy: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        i,
                        total_step,
                        time.time() - starttime,
                        accuracy,
                        sample_loss,
                    )
                )

        logs.append_train_loss([loss_epoch / total_step])
        logs.create_log(model, epoch=epoch, accuracy=accuracy)


def test(opt, phone_dict, context_model, model):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for idx, k in enumerate(test_dataset.file_list):

            audio, filename = test_dataset.get_full_size_test_item(idx)

            model.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)
            model_input = torch.unsqueeze(model_input, 0)

            targets = torch.LongTensor(phone_dict[filename])

            with torch.no_grad():
                for idx, layer in enumerate(context_model.module.fullmodel):
                    if idx + 1 < len(context_model.module.fullmodel):
                        _, z = layer.get_latents(model_input, calc_autoregressive=False)
                        model_input = z.permute(0, 2, 1)
                context, _ = context_model.module.fullmodel[idx].get_latents(
                    model_input, calc_autoregressive=True
                )

                context = context.detach()

                targets = targets.to(opt.device).reshape(-1)
                inputs = context.reshape(-1, n_features)

                # forward pass
                output = model(inputs)

            output = output[-targets.size(0) :]

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if idx % 1000 == 0:
                print(
                    "Step [{}/{}], Accuracy: {:.4f}".format(
                        idx, len(test_dataset.file_list), correct / total
                    )
                )

    accuracy = (correct / total) * 100
    print("Final Testing Accuracy: ", accuracy)
    return accuracy


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt, add_path_var="linear_model")

    opt.batch_size = 8
    opt.num_epochs = 20

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load self-supervised GIM model
    context_model, _ = load_audio_model.load_model_and_optimizer(opt, reload_model=True)

    if opt.model_type != 2:  # == 2 trains a fully supervised model
        context_model.eval()

    # 41 different phones to differentiate
    n_classes = 41
    n_features = context_model.module.reg_hidden

    # create linear classifier
    model = torch.nn.Sequential(torch.nn.Linear(n_features, n_classes)).to(opt.device)
    model.apply(weights_init)

    criterion = torch.nn.CrossEntropyLoss()

    if opt.model_type == 2:
        params = list(context_model.parameters()) + list(model.parameters())
    else:
        params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=1e-4)

    # load dataset
    phone_dict = phone_dict.load_phone_dict(opt)
    _, train_dataset, _, test_dataset = get_dataloader.get_libri_dataloaders(opt)

    logs = logger.Logger(opt)
    accuracy = 0

    try:
        # Train the model
        train(opt, phone_dict, context_model, model)

        # Test the model
        accuracy = test(opt, phone_dict, context_model, model)

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")

    logs.create_log(model, accuracy=accuracy, final_test=True)

    if opt.model_type == 2:
        print("Saving supervised model")
        torch.save(
            context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
        )
