echo "Downloading datasets:"
mkdir datasets
cd datasets || exit
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz || exit
mkdir LibriSpeech100_labels_split
cd LibriSpeech100_labels_split || exit
gdown https://drive.google.com/uc?id=1vSHmncPsRY7VWWAd_BtoWs9-fQ5cBrEB # test split
gdown https://drive.google.com/uc?id=1ubREoLQu47_ZDn39YWv1wvVPe2ZlIZAb # train split
gdown https://drive.google.com/uc?id=1bLuDkapGBERG_VYPS7fNZl5GXsQ9z3p2 # converted_aligned_phones.zip
unzip converted_aligned_phones.zip
cd ../..

