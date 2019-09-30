import torch
import time
import numpy as np

## own modules
from GreedyInfoMax.audio.data import get_dataloader
from GreedyInfoMax.utils import logger, utils
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model


def train(opt, context_model, model):
    total_step = len(train_loader)

    speaker_id_dict = {}
    for idx, key in enumerate(train_dataset.speaker_dict):
        speaker_id_dict[key] = idx

    for epoch in range(opt.num_epochs):
        loss_epoch = 0
        for i, (audio, _, speaker_id, _) in enumerate(train_loader):
            starttime = time.time()

            model_input = audio.to(opt.device)

            if opt.model_type == 2:  ##fully supervised training
                for idx, layer in enumerate(context_model.module.fullmodel):
                    context, _ = layer.get_latents(model_input)
                    model_input = context.permute(0, 2, 1)
            else:
                with torch.no_grad():
                    for l_idx, layer in enumerate(context_model.module.fullmodel):
                        context, z = layer.get_latents(model_input)
                        model_input = z.permute(0, 2, 1)
                context = context.detach()

            inputs = context.reshape(opt.batch_size_multiGPU, -1)

            targets = torch.zeros(opt.batch_size_multiGPU).long()
            for idx, cur_speaker in enumerate(speaker_id):
                targets[idx] = speaker_id_dict[cur_speaker]

            targets = targets.to(opt.device)

            # forward pass
            output = model(inputs)
            loss = criterion(output, targets)

            # calculate accuracy
            accuracy, = utils.accuracy(output.data, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sample_loss = loss.item()

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
                starttime = time.time()

            loss_epoch += sample_loss
        logs.append_train_loss([loss_epoch / total_step])

    return speaker_id_dict


def test(opt, speaker_id_dict, data_loader):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, (audio, filename, speaker_id, _) in enumerate(data_loader):

            # get latent representations for current audio
            model_input = audio.to(opt.device)

            with torch.no_grad():
                for l_idx, layer in enumerate(context_model.module.fullmodel):
                    context, z = layer.get_latents(model_input)
                    model_input = z.permute(0, 2, 1)

            context = context.detach()
            inputs = context.reshape(opt.batch_size_multiGPU, -1)

            targets = torch.zeros(opt.batch_size_multiGPU).long()
            for idx, cur_speaker in enumerate(speaker_id):
                targets[idx] = speaker_id_dict[cur_speaker]

            targets = targets.to(opt.device)

            # forward pass
            output = model(inputs)

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print("Final Testing Accuracy: ", accuracy)
    return accuracy


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt, add_path_var="linear_model")
    opt.num_epochs = 50

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load pretrained model (if model_type == 0)
    context_model, _ = load_audio_model.load_model_and_optimizer(opt, reload_model=True)

    if opt.model_type != 2:
        context_model.eval()

    # input dimensions for linear model
    n_features = int(context_model.module.reg_hidden * np.round(20480 / 160))

    # 41 different phones to differentiate
    n_classes = 251

    # create linear classifier
    model = torch.nn.Sequential(torch.nn.Linear(n_features, n_classes)).to(opt.device)
    criterion = torch.nn.CrossEntropyLoss()

    if opt.model_type == 2:
        params = list(context_model.parameters()) + list(model.parameters())
    else:
        params = model.parameters()

    optimizer = torch.optim.Adam(params)

    # load datasets and loaders
    train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_libri_dataloaders(
        opt
    )

    logs = logger.Logger(opt)

    try:
        # Train the model
        speaker_id_dict = train(opt, context_model, model)

        # Test the model
        accuracy = test(opt, speaker_id_dict, test_loader)

    except KeyboardInterrupt:
        print("Training got interrupted")

    logs.create_log(model, accuracy=accuracy, final_test=True)
