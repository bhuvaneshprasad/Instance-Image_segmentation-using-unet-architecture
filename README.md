# Instance-Image_segmentation-using-unet-architecture
An instance image segmentation deep learning project using U-NET architecture.

## Results
Class           F1         Jaccard   
-----------------------------------
Background     : 0.91726 - 0.85489
Hat            : 0.17345 - 0.14142
Hair           : 0.70944 - 0.60189
Glove          : 0.00000 - 0.00000
Sunglasses     : 0.00000 - 0.00000
UpperClothes   : 0.54907 - 0.44535
Dress          : 0.04839 - 0.03812
Coat           : 0.29541 - 0.24208
Socks          : 0.00000 - 0.00000
Pants          : 0.40370 - 0.33231
Torso-skin     : 0.58930 - 0.46888
Scarf          : 0.00258 - 0.00158
Skirt          : 0.02384 - 0.01836
Face           : 0.80208 - 0.70738
Left-arm       : 0.35982 - 0.27104
Right-arm      : 0.40433 - 0.30608
Left-leg       : 0.06772 - 0.04761
Right-leg      : 0.05562 - 0.03821
Left-shoe      : 0.05771 - 0.03415
Right-shoe     : 0.07405 - 0.04564
-----------------------------------
Mean           : 0.27669 - 0.22975

The results are not great as the classes with fewer pixels has poor f1 and jaccard scores.

The model needs to be retrained by cleaning the dataset and augmenting it.
