import os
from skimage.io import imread
import numpy as np
import cv2

def get_train_data(train_images_dir, train_labels_dir, img_h, img_w, N_channels=3, C=3, gt_list = None):
    print('-'*30)
    print('Loading train images...')
    print('-'*30)
    assert(C == 2 or C > 2)
    files = os.listdir(train_images_dir)  # get file names
    total_images = np.zeros([len(files), img_h, img_w, N_channels])  # for storing training imgs
    for idx in range(len(files)):  #
        img = imread(os.path.join(train_images_dir, files[idx]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img=cv2.resize(img,(img_h,img_h),interpolation=cv2.INTER_CUBIC)
        total_images[idx, :, :, :img.shape[-1]] = img
    total_images = total_images/np.max(total_images)
    mean = np.mean(total_images, axis=0)
    # np.save('./mean_img.npy', mean)
    total_images -= mean
    total_images = np.transpose(total_images, [0, 3, 1, 2])

    print('-'*30)
    print('Loading train labels...')
    print('-'*30)
    files2 = os.listdir(train_labels_dir)
    total_labels = np.zeros([len(files), img_h, img_w, C])
    if C == 2:
        for idx in range(len(files)):
            ground_truth = imread(os.path.join(train_labels_dir, files2[idx]))
            total_labels[idx, :, :, 0] = ((ground_truth == 0)*1)
            total_labels[idx, :, :, 1] = ((ground_truth != 0)*1)
    else:
        if gt_list is None:
            print('-' * 30 + '\n' + 'There is a lack of a list of GT values!')
            raise Exception
        else:
            for idx in range(len(files)):
                mask = imread(os.path.join(train_labels_dir, files2[idx]))
                mask_ = mask
                masks = np.where(mask_ < 128, 0, mask_)
                # print('train mask unique',np.unique(masks))
                masks=cv2.resize(masks,(img_h,img_h),interpolation=cv2.INTER_CUBIC)
                for ch in range(C):
                    total_labels[idx, :, :, ch] = ((masks == gt_list[ch]) * 1)  # onehot
                # print('total_labelse',np.unique(total_labels))

    return total_images, total_labels


def get_test_data(test_images_dir, test_labels_dir, img_h, img_w, N_channels=3, C=3):
    print('-'*30)
    print('Loading test images...')

    files = os.listdir(test_images_dir)  # get file names
    total_images = np.zeros([len(files), img_h, img_w, N_channels])
    for idx in range(len(files)):
        img = imread(os.path.join(test_images_dir,files[idx]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img=cv2.resize(img,(img_h,img_h),interpolation=cv2.INTER_CUBIC)
        total_images[idx, :, :, :img.shape[-1]] = img
    total_images = total_images/np.max(total_images)
    mean = np.load('./mean_img.npy')
    total_images -= mean
    total_images = np.transpose(total_images, [0, 3, 1, 2])  # transpose to shape[N, channels, h, w]

    print('-'*30)
    print('Loading test labels...')
    print('-'*30)
    files2 = os.listdir(test_labels_dir)
    total_labels = np.zeros([len(files2), img_h, img_w])
    for idx in range(len(files2)):
        mask = imread(os.path.join(test_labels_dir, files2[idx]))
        mask_ = mask
        total_labels = np.where(mask_ < 128, 0, mask_)
        total_labels=cv2.resize(total_labels,(img_h,img_h),interpolation=cv2.INTER_CUBIC)

    return total_images, total_labels


def pred_to_imgs(predictions, img_h, img_w, C=3):
    assert(len(predictions.shape) == 3)
    assert(predictions.shape[1] == img_h*img_w)
    N_images = predictions.shape[0]
    predictions = np.reshape(predictions, [N_images, img_h, img_w, C])
    pred_images = np.argmax(predictions, axis=3)
    pred_images = pred_images.astype(np.float)
    return pred_images


def pred_to_imgs_bak(predictions, img_h, img_w, C=3):  # 和pred_to_imgs应该是一样的，以另一种方式做的argmax
    assert(len(predictions.shape) == 3)
    assert(predictions.shape[1] == img_h*img_w)
    N_images = predictions.shape[0]
    predictions = np.reshape(predictions,[N_images, img_h, img_w, C])
    pred_images = np.zeros([N_images, img_h, img_w])
    for img in range(N_images):
        for h in range(img_h):
            for w in range(img_w):
                l = list(predictions[img, h, w, :])
                pred_images[img, h, w] = l.index(max(l))  # float
    pred_images /= np.max(pred_images)
    assert(np.min(pred_images) == 0 and np.max(pred_images) == 1)

    return pred_images











