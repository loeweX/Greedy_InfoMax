#!/bin/sh

echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -m GreedyInfoMax.vision.main_vision --download_dataset --save_dir vision_experiment

echo "Testing the Greedy InfoMax Model for image classification"
python -m GreedyInfoMax.vision.downstream_classification --model_path ./logs/vision_experiment --model_num 299