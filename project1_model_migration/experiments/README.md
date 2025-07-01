# Experiment Record: resnet50_bs32_lr0.001_e10 (branch)

## Basic Information
- **Framework**: TensorFlow (TF) & PyTorch
- **Model**: ResNet50 (all convolutional layers frozen) + 102-way classifier
- **Classification Task**: Oxford Flowers 102
- **Input Size**: 224x224
- **Number of Classes**: 102
- **Training Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.001

## Model Architecture
```text
ResNet50 (pretrained weights, include_top=False)
→ GlobalAveragePooling2D
→ Dense(128, relu)
→ Dropout(0.5)
→ Dense(102, softmax)
