# UNET_Multi
 U-Net for multi-class segmentation with tf2.0 
 Original code from: https://github.com/fosaken/U-Net-segmentation
 Dataset from:https://github.com/zxaoyou/segmentation_WBC
 
 ---
 这个代码只是单纯的想要尝试多类分割，所以效果不好，只是能分出3个类别来而已
 多类分割Ground Truth需要进行one-hot编码
 激活函数一般用softmax，loss用ce，output需要用到argmax
 
 至于loss用dice的话应该怎么做我还不知道
 
