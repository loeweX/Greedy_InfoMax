import pickle
import os

def load_phone_dict(opt):
    dir_name = os.path.join(opt.data_input_dir, "Phone_dict/")
    filename = os.path.join(dir_name, "phone_dict.pkl")

    if not os.path.exists(filename):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return create_dict_from_phones(
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech100_labels_split/converted_aligned_phones.txt",
            ),
            filename,
        )
    else:
        with open(filename, "rb") as f:
            return pickle.load(f)


def save_obj(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def create_dict_from_phones(phone_path, save_path):
    phone_dict = {}
    print("Creating phone dictionary")
    with open(phone_path, "r") as rf:
        for idx, line in enumerate(rf.readlines()):
            tmp = line.replace("\n", "").split(" ")
            sample_id = tmp[0]
            phones = [int(i) for i in tmp[1:]]
            phone_dict[sample_id] = phones
            if idx % 1000 == 0:
                print("..")

    save_obj(phone_dict, save_path)
    return phone_dict
