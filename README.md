# Plant-Leaf-Detection-system
 Plant Leaf Classification System
A high-accuracy deep learning solution for identifying plant leaf species and health status. Built with TensorFlow and Transfer Learning, this modular project achieves ≥99% top-1 accuracy on the 38-class PlantVillage dataset.

* Features
  
EfficientNetB3 (customizable to ResNet50 or DenseNet121) for powerful feature extraction.

Attention Mechanism
Channel-wise attention block to emphasize critical leaf regions.

Extensive Data Augmentation
Random flips, rotations, zooms, brightness adjustments for robust generalization.

Early Stopping & Checkpointing
Prevents overfitting and automatically saves the best model.

Single-Image Prediction
predict_single_image() helper for instant inference on new leaf photos.

Comprehensive Evaluation
Test-set accuracy, classification report, and normalized confusion matrix.
