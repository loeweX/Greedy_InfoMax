from GreedyInfoMax.audio.linear_classifiers import SVM_speaker

def val_by_SVM_speaker_classification(opt, model, train_dataset, test_dataset, SVM_num_speakers=20):
    model.eval()
    opt.SVM_num_speakers = SVM_num_speakers

    print("##### Validating by SVM training for Speakers #####")

    # Train the SVM
    speaker_svm, speaker_id_dict, training_speaker_keys = SVM_speaker.SVM_train(opt, model, train_dataset)

    # Test the SVM
    accuracy = SVM_speaker.SVM_test(opt, speaker_id_dict, model, speaker_svm, training_speaker_keys, test_dataset)

    model.train()
    return accuracy