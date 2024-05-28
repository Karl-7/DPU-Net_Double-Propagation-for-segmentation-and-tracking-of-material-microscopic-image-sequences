this project can help you run 3 models including U-Net, WPU-Net, and DPU-net. Please notice that DPU-Net still has the process of weighting the loss, which is used in the WPU-Net, it's just not in its name(well you can call it DWPU-Net anyway, I just named it this way cuz its easier to read hehehe).

to run the model ,you can add a if name=="__main__" progress in the bottom of TRAIN_SETTINGS.py. inside this .py file, you can find all 3 functions: run_train. run_test, and run_metric.

the general hyperparameters can be defined in TRAIN_SETTINGS.py as well. although you can find them in TRAIN_PARAS.py, I gave up on the idea of creating a useless new class in the end, and the program settings in TRAIN_SETTINGS will automatically cover up the original settings in TRAIN_PARAS anyway.

please notice that your dataset should be arranged in the following format:
<img width="96" alt="image" src="https://github.com/Karl-7/Double-Propagation-Net-for-segmentation-and-tracking-of-material-microscopic-image-sequences-/assets/142679657/94cf6af1-6d93-4a9b-9685-215619050420">
