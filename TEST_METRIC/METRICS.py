import math
import numpy as np
from skimage import metrics, measure, color
from skimage.measure import label
import EVALUATE as ev
from typing import Tuple
import matplotlib.pyplot as plt
import decimal


# 我们将图像分割指标分成5个类别：
# 比如：基于像素的，基于类内重合度的，基于边缘的，基于聚类的 和 基于实例的
# We grouped image segmentation metrics into five groups:
# Such as pixel based, region based, boundary based, clustering based, instance based

# 注意：
# 对于下列所有方法，pred是分割结果，mask是真值，所有像素的值为整数，且在[0,C], C为分割类别
# Note:
# For all the metrics below, pred is the segmentation result of certain method and
# mask is the ground truth. The value is integer and range from [0, C], where C is
# class of segmentation


def get_metric(metric_name: str, output: np.array, label: np.array) -> float:
    """
        Evaluate output and label and get metric value during training
    """
    temp_metric = 0.0
    if metric_name == 'dice':
        temp_metric = get_dice(output, label)
    elif metric_name == 'vi':
        _, _, temp_metric = get_vi(output, label)
    elif metric_name == 'ARI':
        temp_metric = get_ari(output, label)
    elif metric_name == 'MAP':
        temp_metric = get_map_2018kdsb(output, label)
    elif metric_name == 'GRAIN':  # 晶粒度
        temp_metric = GrainSize(output)
    return temp_metric


def GrainSize(img, M=100):
    """
    计算奥氏体平均晶粒度G
    M：放大倍数
    L：截线长度
    P_avg:平均截点数
    计算公式 G = 6.643856 * lg((M*P_avg)/L) - 3.288
    return:奥氏体平均晶粒度G
    """
    row_length = img.shape[0]  # 宽
    col_length = img.shape[1]  # 长
    print(row_length, col_length)
    L = 0.603888
    list_row = []
    # 三条截线初始位置(这里是竖着划线，截断长的边
    for i in range(3):
        temp = row_length * (i + 1) * 0.25
        list_row.append(int(temp))
    # 获取截点
    P = [0, 0, 0]
    for i in range(3):
        count = 0
        for j in img[list_row[i]]:
            # 截断长边的时候
            # if img[count][list_row[i]] != img[count + 1][list_row[i]]:
            # 截断短边的时候
            if img[list_row[i]][count] != img[list_row[i]][count + 1]:
                P[i] += 1
            count += 1
            if count == col_length - 1:
                break
    P_avg = (P[0] + P[1] + P[2]) / (3 * 2)
    #  计算晶粒度G
    G = 6.643856 * math.log((M * P_avg) / L, 10) - 3.288
    print("计算得到的晶粒度：", G)
    return math.fabs(G)

        # G = decimal.Decimal(G).quantize(decimal.Decimal("0.0001"))、
    
def get_total_evaluation(pred: np.ndarray, mask: np.ndarray, require_edge: bool = True) -> Tuple:
    """
         Get whole evaluation of all metrics
         Return Tuple, (value_list, name_list)
    """
    gala = get_vi(pred, mask)
    if require_edge:
        metric_values = [get_pixel_accuracy(pred, mask), get_mean_accuracy(pred, mask),
                         get_iou(pred, mask), get_fwiou(pred, mask), get_dice(pred, mask),
                         get_figure_of_merit(pred, mask), get_completeness(pred, mask), get_correctness(pred, mask),
                         get_quality(pred, mask),
                         get_ri(pred, mask), get_ari(pred, mask), gala[0], gala[1], gala[2],
                         get_cardinality_difference(pred, mask), get_map_2018kdsb(pred, mask)]
    else:
        metric_values = [get_pixel_accuracy(pred, mask), get_mean_accuracy(pred, mask),
                         get_iou(pred, mask), get_fwiou(pred, mask), get_dice(pred, mask),
                         get_ri(pred, mask), get_ari(pred, mask), gala[0], gala[1], gala[2],
                         get_cardinality_difference(pred, mask), get_map_2018kdsb(pred, mask)]
    return metric_values


# ************** 基于像素的评估 Pixel based evaluation **************
# pixel accuracy, mean accuracy
def get_pixel_accuracy(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Pixel accuracy for whole image
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1

    temp_n_ii = 0.0
    temp_t_i = 0.0
    for i_cl in range(class_num):
        temp_n_ii += np.count_nonzero(mask[pred == i_cl] == i_cl)
        temp_t_i += np.count_nonzero(mask == i_cl)
    value = temp_n_ii / temp_t_i
    return value


def get_mean_accuracy(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean accuracy for each class
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1
    temp = 0.0
    for i_cl in range(class_num):
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl)
        temp += n_ii / t_i
    value = temp / class_num
    return value


# ************** 基于类内重合度的评估 Region based evaluation **************
# Mean IOU (mIOU), Frequency weighted IOU(FwIOU), Dice score
def get_iou(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Referenced by:
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1

    temp = 0.0
    for i_cl in range(class_num):
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl)
        temp += n_ii / (t_i + np.count_nonzero(pred == i_cl) - n_ii)
    value = temp / class_num
    return value


def get_fwiou(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Referenced by:
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1

    temp_t_i = 0.0
    temp_iou = 0.0
    for i_cl in range(0, class_num):
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl)
        temp_iou += t_i * n_ii / (t_i + np.count_nonzero(pred == i_cl) - n_ii)
        temp_t_i += t_i
    value = temp_iou / temp_t_i
    return value


def get_dice(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Dice score
    From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
    """
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1), plt.imshow(pred, cmap="gray"), plt.title("pred"), plt.axis("off")
    # plt.subplot(1, 3, 2), plt.imshow(label, cmap="gray"), plt.title("label"), plt.axis("off")
    plt.subplot(1, 3, 3), plt.imshow(mask, cmap="gray"), plt.title("mask"), plt.axis("off")

    # plt.savefig(str(Path(test_vis_path, img_name)))
    plt.show()
    # intersection = np.count_nonzero(mask[pred == 1] == 1)
    # area_sum = np.count_nonzero(mask == 1) + np.count_nonzero(pred == 1)
    intersection = np.count_nonzero(mask[pred == 255] == 255)
    area_sum = np.count_nonzero(mask == 255) + np.count_nonzero(pred == 255)
    value = 2 * intersection / area_sum
    return value


# ************** 基于边缘的评估 boundary based evaluation **************
# figure of merit
def get_figure_of_merit(pred: np.ndarray, mask: np.ndarray, boundary_value: int = 0, const_index: float = 0.1) -> float:
    """
    Referenced by:
    Abdou I E, Pratt W K. Quantitative design and evaluation of enhancement thresholding edge detectors[J].
    Proceedings of the IEEE, 1979, 67(5): 753-763
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    num_max = num_pred if num_pred > num_mask else num_mask
    temp = 0.0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                temp = temp + 1 / (1 + const_index * pow(distance, 2))
    f_score = (1.0 / num_max) * temp
    return f_score


def get_dis_from_mask_point(mask: np.ndarray, index_x: int, index_y: int, boundary_value: int = 0,
                            neighbor_length: int = 20):
    """
    Calculation the distance between the boundary point(pred) and its nearest boundary point(mask)
    """

    if mask[index_x, index_y] == 255:
        return 0
    distance = neighbor_length / 2
    region_start_row = 0
    region_start_col = 0
    region_end_row = mask.shape[0]
    region_end_col = mask.shape[1]
    if index_x - neighbor_length > 0:
        region_start_row = index_x - neighbor_length
    if index_x + neighbor_length < mask.shape[0]:
        region_end_row = index_x + neighbor_length
    if index_y - neighbor_length > 0:
        region_start_col = index_y - neighbor_length
    if index_y + neighbor_length < mask.shape[1]:
        region_end_col = index_y + neighbor_length
        # Get the corrdinate of mask in neighbor region
        # becuase the corrdinate will be chaneged after slice operation, we add it manually
    x, y = np.where(mask[region_start_row: region_end_row, region_start_col: region_end_col] == boundary_value)
    try:
        min_distance = np.amin(
            np.linalg.norm(np.array([x + region_start_row, y + region_start_col]) - np.array([[index_x], [index_y]]),
                           axis=0))
        return min_distance
    except ValueError as e:
        return neighbor_length


# completeness
def get_completeness(pred: np.ndarray, mask: np.ndarray, theta: float = 2.0, boundary_value: int = 0) -> float:
    """
    Referenced by:
    Beyond the Pixel-Wise Loss for Topology-Aware Delineation
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    temp_pred_mask = 0
    for index_x in range(0, mask.shape[0]):
        for index_y in range(0, mask.shape[1]):
            if mask[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(pred, index_x, index_y)
                if distance < theta:
                    temp_pred_mask += 1
    f_score = float(temp_pred_mask) / float(num_mask)
    return f_score


# correctness
def get_correctness(pred: np.ndarray, mask: np.ndarray, theta: float = 2.0, boundary_value: int = 0) -> float:
    """
    Referenced by:
    Beyond the Pixel-Wise Loss for Topology-Aware Delineation
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    temp_mask_pred = 0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                if distance < theta:
                    temp_mask_pred += 1
    f_score = float(temp_mask_pred) / float(num_pred)
    return f_score


# quality
def get_quality(pred: np.ndarray, mask: np.ndarray, theta: float = 2.0, boundary_value: int = 0) -> float:
    """
    Referenced by:
    Beyond the Pixel-Wise Loss for Topology-Aware Delineation
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    temp_pred_mask = 0
    for index_x in range(0, mask.shape[0]):
        for index_y in range(0, mask.shape[1]):
            if mask[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(pred, index_x, index_y)
                if distance < theta:
                    temp_pred_mask += 1
    temp_mask_pred = 0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                if distance < theta:
                    temp_mask_pred += 1
    f_score = float(temp_mask_pred) / float(num_pred - temp_pred_mask + num_mask)
    return f_score


# ************** 基于聚类的评估 Clustering based evaluation **************
# Rand Index (RI), Adjusted Rand Index (ARI) and Variation of Information (VI)
def get_ri(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    Rand index
    Implemented by gala (https://github.com/janelia-flyem/gala.)
    """
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    value = ev.rand_index(label_pred, label_mask)
    return value


def get_ari(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    Adjusted rand index
    Implemented by gala (https://github.com/janelia-flyem/gala.)
    """
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    value = ev.adj_rand_index(label_pred, label_mask)
    return value


def get_vi(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0, method: int = 1) -> Tuple:
    """
    Referenced by:
    Marina Meilă (2007), Comparing clusterings—an information based distance,
    Journal of Multivariate Analysis, Volume 98, Issue 5, Pages 873-895, ISSN 0047-259X, DOI:10.1016/j.jmva.2006.11.013.
    :param method: 0: skimage implementation and 1: gala implementation (https://github.com/janelia-flyem/gala.)
    :return Tuple = (VI, merger_error, split_error)
    """
    vi, merger_error, split_error = 0.0, 0.0, 0.0

    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    if method == 0:
        # scikit-image
        split_error, merger_error = metrics.variation_of_information(label_mask, label_pred)
    elif method == 1:
        # gala
        merger_error, split_error = ev.split_vi(label_pred, label_mask)
    vi = merger_error + split_error
    if math.isnan(vi):
        return 10, 5, 5
    return merger_error, split_error, vi


# ************** 基于实例的评估 Instance based evaluation **************
# cardinality difference, MAP
def get_cardinality_difference(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
    R = |G| - |P|
    |G| is number of region in mask, and |P| is number of region in pred
    R > 0 refers to under segmentation and R < 0 refers to over segmentation
    Referenced by
    Waggoner J , Zhou Y , Simmons J , et al. 3D Materials Image Segmentation by 2D Propagation: A Graph-Cut Approach Considering Homomorphism[J].
    IEEE Transactions on Image Processing, 2013, 22(12):5282-5293.
    """

    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    value = num_mask - num_pred
    return value * 1.0


def get_map_2018kdsb(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    Mean Average Precision
    From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
    Referenced from 2018 kaggle data science bowl:N
    https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    """
    # 阈值范围从0.5到0.95
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)

    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    label_mask_show = color.label2rgb(label_mask)
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_pred_show = color.label2rgb(label_pred)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # ax1.imshow(label_mask_show, interpolation='nearest')
    # ax1.axis('off')
    # ax2.imshow(label_pred_show, interpolation='nearest')
    # ax2.axis('off')

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1), plt.imshow(label_mask_show), plt.title("mask"), plt.axis("off")
    plt.subplot(1, 2, 2), plt.imshow(label_pred_show), plt.title("label"), plt.axis("off")
    # plt.subplot(1, 3, 3), plt.imshow(output, cmap="gray"), plt.title("output"), plt.axis("off")

    # plt.savefig(str(Path(test_vis_path, img_name)))
    plt.show()

    # print("num_pred", num_pred)
    # print("num_mask", num_mask)
    # 默认不计算背景的ap值
    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))  # 获得与之相交的所有label
        # 对与其相交的的所有mask label计算iou，后取其最值
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:  # 如果pred的某一个label没有与之对应的mask的label,则继续下一个label
            continue

        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))
        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        # 根据最值将tp赋值
        # Assumption: There is only a region whose IOU > 0.5 for target region
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    print(tp)
    fp = num_pred - tp
    fn = num_mask - tp
    tn = 0
    # print("Precision: ", np.average(tp / (tp + fp)))
    # print("Recall: ", np.average(tp / (tp + fn)))
    # print("Accuracy: ", np.average((tp + tn) / (tp + fp + tn + fn)))
    value = np.average(tp / (tp + fp + fn))
    return value


def get_map_2018kdsb_binary(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    Mean Average Precision
    From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
    Referenced from 2018 kaggle data science bowl:N
    https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    """
    # 阈值范围从0.5到0.95
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)

# 因为只有两个图像，所以不进行label也可以，只看是实际为1的区域的重合程度
#     label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    # label_mask_show = color.label2rgb(label_mask)
    # label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    # label_pred_show = color.label2rgb(label_pred)
    # 写死label
    label_mask = mask
    num_mask = 1
    label_pred = pred
    num_pred = 1
    # # 展示label效果
    # plt.figure(figsize=(20, 20))
    # plt.subplot(1, 2, 1), plt.imshow(label_mask_show), plt.title("mask"), plt.axis("off")
    # plt.subplot(1, 2, 2), plt.imshow(label_pred_show), plt.title("label"), plt.axis("off")
    # plt.show()

    # 默认不计算背景的ap值，因为这个数据里面label只有0和1两个值
    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))  # 获得与之相交的所有label的值
        # 对与其相交的的所有mask label计算iou，后取其最值
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:  # 如果pred的某一个label没有与之对应的mask的label,则继续下一个label
            continue

        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))
        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        # 根据最值将tp赋值
        # Assumption: There is only a region whose IOU > 0.5 for target region
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    print('tp:', tp)
    fp = num_pred - tp
    fn = num_mask - tp
    tn = 0
    print("Precision: ", np.average(tp / (tp + fp)))
    print("Recall: ", np.average(tp / (tp + fn)))
    print("Accuracy: ", np.average((tp + tn) / (tp + fp + tn + fn)))
    value = np.average(tp / (tp + fp + fn))
    return value
