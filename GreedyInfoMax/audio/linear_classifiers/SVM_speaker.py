import torch
import numpy as np
from sklearn import svm
import time

## own modules
from GreedyInfoMax.audio.data import get_dataloader
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model


def SVM_train(opt, context_model, train_dataset, SVM_num_speakers=20):
    """
    train an SVM on the representation created by the GIM model
    :param context_model: GIM model
    :param SVM_num_speakers: number of speakers to include for SVM training and testing (default: 20)
    """
    speaker_id_dict = {}
    for idx, key in enumerate(train_dataset.speaker_dict):
        speaker_id_dict[key] = idx

    svm_targets = [None] * opt.SVM_training_samples * SVM_num_speakers
    svm_inputs = [None] * opt.SVM_training_samples * SVM_num_speakers

    training_speaker_keys = [None] * SVM_num_speakers

    starttime = time.time()

    speaker_idx = 0
    for idx, k in enumerate(train_dataset.speaker_dict):

        if speaker_idx == SVM_num_speakers:
            break

        audio = train_dataset.get_audio_by_speaker(
            k, batch_size=opt.SVM_training_samples
        )

        # skip speakers that do not have enough training samples
        if audio.size(0) < opt.SVM_training_samples:
            continue

        model_input = audio.to(opt.device)

        with torch.no_grad():
            for _, layer in enumerate(context_model.module.fullmodel):
                context, _ = layer.get_latents(model_input)
                model_input = context.permute(0, 2, 1).detach()
        context = context.detach()

        svm_targets[
            idx * opt.SVM_training_samples : idx * opt.SVM_training_samples
            + opt.SVM_training_samples
        ] = [speaker_id_dict[k]] * opt.SVM_training_samples

        inputs = context.reshape(opt.SVM_training_samples, -1)
        svm_inputs[
            idx * opt.SVM_training_samples : idx * opt.SVM_training_samples
            + opt.SVM_training_samples
        ] = (inputs.cpu().numpy().tolist())

        training_speaker_keys[speaker_idx] = k
        speaker_idx += 1

    speaker_svm = svm.SVC(
        gamma="scale", decision_function_shape="ovo", cache_size=1000, kernel="linear"
    )
    speaker_svm.fit(svm_inputs, svm_targets)

    print("Time: ", time.time() - starttime)

    return speaker_svm, speaker_id_dict, training_speaker_keys


def SVM_test(
    opt,
    speaker_id_dict,
    context_model,
    speaker_svm,
    training_speaker_keys,
    test_dataset,
    num_samples_per_speaker=10,
):
    total = 0
    correct = 0

    speaker_idx = 0

    # loop over training_speaker_keys to make sure that same speakers are used for training and testing
    for idx, k in enumerate(training_speaker_keys):

        audio = test_dataset.get_audio_by_speaker(k, batch_size=num_samples_per_speaker)

        if audio.size(0) < num_samples_per_speaker:
            continue

        model_input = audio.to(opt.device)

        with torch.no_grad():
            for _, layer in enumerate(context_model.module.fullmodel):
                context, _ = layer.get_latents(model_input)
                model_input = context.permute(0, 2, 1).detach()
        context = context.detach()

        target = speaker_id_dict[k]

        inputs = context.reshape(num_samples_per_speaker, -1)
        svm_input = inputs.cpu().numpy().tolist()

        svm_output = speaker_svm.predict(svm_input)

        # calculate accuracy
        total += num_samples_per_speaker
        correct += (svm_output == target).sum()

        speaker_idx += 1

    accuracy = correct / total
    print("Testing Accuracy SVM: ", accuracy)
    return accuracy


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt, add_path_var="linear_model")
    opt.SVM_training_samples = 20

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load pretrained model
    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt, reload_model=True
    )
    context_model.eval()

    _, train_dataset, _, test_dataset = get_dataloader.get_libri_dataloaders(opt)

    try:
        # Train the model
        speaker_svm, speaker_id_dict, training_speaker_keys = SVM_train(
            opt, context_model, train_dataset
        )

        # Test the model
        accuracy = SVM_test(
            opt,
            speaker_id_dict,
            context_model,
            speaker_svm,
            training_speaker_keys,
            test_dataset,
        )

    except KeyboardInterrupt:
        print("Training got interrupted")
