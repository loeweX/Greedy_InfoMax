import torch
import numpy as np

from GreedyInfoMax.utils import utils


def val_by_latent_speakers(opt, dataset, model, epoch, step):
    """
    Validate the training process by plotting the t-SNE
    representation of the latent space for different speakers
    """
    big_feature_space = []
    max_speakers = 10
    batch_size = 10

    input_size = (opt.batch_size, 1, 20480)

    model.eval()
    for idx, layer in enumerate(model.module.fullmodel):
        latent_rep_size, latent_rep_length = layer.get_latent_seq_len(input_size)
        big_feature_space.append(
            np.zeros((max_speakers, batch_size, latent_rep_size * latent_rep_length))
        )
        input_size = (opt.batch_size, layer.enc_hidden, latent_rep_length)

    speaker_labels = np.zeros((len(model.module.fullmodel), max_speakers, batch_size))
    counter = 0

    with torch.no_grad():
        for idx, k in enumerate(dataset.speaker_dict):

            if idx == max_speakers:
                break

            audio = dataset.get_audio_by_speaker(k, batch_size=batch_size)

            if audio.size(0) != batch_size:
                max_speakers += 1
                continue

            model_input = audio.to(opt.device)

            for idx, layer in enumerate(model.module.fullmodel):
                context, z = layer.get_latents(model_input)
                model_input = z.permute(0, 2, 1)

                latent_rep = context.permute(0, 2, 1).cpu().numpy()
                big_feature_space[idx][counter, :, :] = np.reshape(
                    latent_rep, (batch_size, -1)
                )
                speaker_labels[idx, counter, :] = counter

            counter += 1

    for idx, layers in enumerate(model.module.fullmodel):
        utils.fit_TSNE_and_plot(
            opt,
            big_feature_space[idx],
            speaker_labels[idx],
            "{}_{}_model_{}".format(epoch, step, idx),
        )

    model.train()
