import os
import os.path as osp
from pathlib import Path
import re
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.patches import Rectangle
from PIL import Image
import time
# from ..datasets.pipelines.io4med import open_mask_random, convert_label


print_tensor = lambda name, x : print(name, type(x), x.shape, x.dtype, x.min(), x.max())

def fp2store(out_dir :Path, mask_fp:str, suffix: str = "", filename_extension = '.png'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # print('check1:', out_dir)
    name, source = flat_path_id(mask_fp)
    res_dir = out_dir /source
    res_dir.mkdir(parents=True, exist_ok=True)
    res = res_dir/str(f"{name}_{suffix}{filename_extension}")
    # print('check2:', str(res))
    return str(res)


def flat_path_id(path_w_subdir):
    source, pid, studyid, seriesid, slice_index =  path_w_subdir.split(os.sep)[-5:] # name = subdir/###.png
    # print(fp_chunks)
    short_fn = '_'.join([pid, studyid[-6:], seriesid[-6:], slice_index.replace('.png', '')])
    return short_fn, source

class PlotSeg2Image(object):

    def __init__(self, image):
        fig = plt.figure(figsize=(16, 8))
        # plot ct image on the left as reference
        ax_ref = fig.add_subplot(121)
        ax_ref.imshow(image, cmap='gray')
        ax_ref.set_xticks([]), ax_ref.set_yticks([])
        self.fig = fig
        self.ax_ref = ax_ref

        # plot ct image on the right with other contours
        self.ax_mask = fig.add_subplot(122)
        self.ax_mask.imshow(image, cmap='gray')

    def put_on_mask(self, mask, color  = 'r', is_bbox_on = True):
        #skimage findcontours生成的，rows, cols
        contours = measure.find_contours(mask, 0.5, fully_connected='low', positive_orientation='low')
        self.put_on_edge(contours, color, is_bbox_on= is_bbox_on)

    def put_on_edge(self, countours, color = 'r', is_col_first = False, is_bbox_on = True):
        xy_ixs = [0, 1] if is_col_first else [1, 0]
        for contour in countours:
            contour = np.array(contour, dtype=np.int16)
            self.ax_mask.plot(contour[:, xy_ixs[0]], contour[:, xy_ixs[1]], color, linewidth=0.8)

            if is_bbox_on:
                contour = contour[..., ::(1 if is_col_first else -1)]
                r_min, c_min = np.min(contour, axis = 0)
                r_max, c_max = np.max(contour, axis = 0)
                self.put_on_bbox([c_min, r_min, c_max, r_max])

        self.ax_mask.set_xticks([]), self.ax_mask.set_yticks([])
        # return self.ax_mask

    def put_on_bbox(self, bbox, color = 'g'):
        # Create a Rectangle patch
        x1, y1, x2, y2 = bbox
        # lower left
        bottom_left_coord, width, height = (x1, y1), x2-x1,y2-y1
        # print(bottom_left_coord, width, height)
        rect = Rectangle(bottom_left_coord, width, height,linewidth=0.5, edgecolor=color,facecolor='none')
        # Add the patch to the Axes
        self.ax_mask.add_patch(rect)
        return self.ax_mask


    def save_fig(self, pic_save_path):
        self.pic_save_path = pic_save_path
        if pic_save_path is not None:
            plt.savefig(pic_save_path, bbox_inches='tight'), plt.close(self.fig)

def dice_score_binary(targets_i, outputs_i, eps = 1e-7):
    intersection = (targets_i * outputs_i).sum()
    sum_of_2 = targets_i.sum() + outputs_i.sum()
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    dice = (2 * intersection + eps * (sum_of_2 == 0)) / (sum_of_2 + eps)
    return dice

# def load_mask_med(gt_fp, label_mapping = None):
#     return convert_label(open_mask_random(gt_fp), label_mapping)

def plotOneImage(img, cmap = None, title = None):
    plt.figure()
    plt.imshow(img, cmap=plt.get_cmap(cmap))
    if title is not None:
        plt.xlabel(title)
    plt.show()


def plot2Image(img_2d, mask_2d, cmap='viridis',
               fig_title = None, save_dir= None):
    img_3d = [img_2d, mask_2d]
    fig = plt.figure(figsize=(8, 4))
    for i in range(2):
        fig.add_subplot(1, 2, i + 1)
        # plt.xticks([])
        # plt.yticks([])
        plt.grid(False)
        plt.imshow(img_3d[i], cmap=cmap)
    # plt.colorbar()
    if bool(save_dir):
        fig.savefig(os.path.join(save_dir, str(fig_title)))
        plt.close()
    else:
        if fig_title is not None:
            plt.xlabel(str(fig_title))
        plt.show()



def plotNImage(img_2d_list, cmap='viridis', rows = 1,
               title_list = None,
               fig_title = None,
               is_close= False):
    nb_img = len(img_2d_list)
    assert isinstance(cmap, (str, list, tuple))
    if isinstance(cmap, str): cmap = [cmap] * nb_img
    else: assert len(cmap) == nb_img
    if title_list is None: title_list = ['na'] * nb_img
    assert len(title_list) == nb_img
    row_size = rows
    col_size = nb_img //row_size + (1 if (nb_img % row_size) > 0 else 0)
    fig = plt.figure(figsize=(6 * col_size, 6 * row_size))
    # fig, axs = plt.subplots(nrows = row_size, ncols= col_size, figsize=(4 * col_size, 4 * row_size))
    for i in range(nb_img):
        ax = plt.subplot(row_size, col_size, i+1)
        # fig.add_subplot(row_size, col_size, i + 1)
        img_list = img_2d_list[i]
        if not isinstance(img_list, list): img_list = [img_list]
        ax.axis('off')
        ax.grid(False)
        alpha_list = [1 / pow(2, n) for n in range(len(img_list))][::-1]
        for j, img in enumerate(img_list):
            ax.imshow(img, cmap=cmap[i] if j == 0 else 'gray', alpha = alpha_list.pop()) #0 (transparent) and 1 (opaque)
        plt.title('%s[%0.1f~%0.1f]' %(title_list[i], img_list[0].min(), img_list[0].max()), fontsize=16)
        # ax.set_title('%s-%0.3f-%0.3f' %(title_list[i], img.min(), img.max()))
    # plt.colorbar()
    if fig_title is not None:
        # plt.xlabel(str(fig_title))
        fig.suptitle(fig_title,  x=0.5, y=0.2)
    if is_close: plt.close() 
    else: plt.show()

    return fig

def save_fig(fig, fp):
    # plt.show()
    fig.savefig(fp)


# def visContour3view(x_vol, y_vol, pat_id='default', save_path=None,
#                     rows=3, cols=5, start_with=0, show_every=1):
#     base_size = 5
#     fig, ax = plt.subplots(rows, cols, figsize=[cols * base_size, rows * base_size])
#     ##冠状面视图，显示volumes 范围和label所在的层
#     mask_proj_y = np.sum(np.sum(y_vol, axis=0), axis=1)
#     y_ind = np.argmax(mask_proj_y)
#     i = 0
#     ax[int(i / cols), int(i % cols)].set_title('coronal view')
#     ax[int(i / cols), int(i % cols)].imshow(x_vol[:, y_ind, :], cmap=plt.get_cmap('gray'))
#     ax[int(i / cols), int(i % cols)].imshow(y_vol[:, y_ind, :], cmap=plt.get_cmap('gray'), alpha=0.4)
#     ax[int(i / cols), int(i % cols)].axis('off')

#     ##矢状面视图
#     mask_proj_x = np.sum(np.sum(y_vol, axis=0), axis=0)
#     x_ind = np.argmax(mask_proj_x)
#     i = 1
#     ax[int(i / cols), int(i % cols)].set_title('sagtiial view')
#     ax[int(i / cols), int(i % cols)].imshow(x_vol[:, :, x_ind], cmap=plt.get_cmap('gray'))
#     ax[int(i / cols), int(i % cols)].imshow(y_vol[:, :, x_ind], cmap=plt.get_cmap('gray'), alpha=0.4)
#     ax[int(i / cols), int(i % cols)].axis('off')

#     ##横截面
#     label_ind = [a for a in range(y_vol.shape[0]) if np.amax(y_vol[a]) > 0]
#     ind_target = label_ind + list(set(range(x_vol.shape[0])).difference(set(label_ind)))
#     img_roi = x_vol[ind_target]
#     label_roi = y_vol[ind_target]

#     exist_label = np.unique(y_vol)[1:]
#     # num_organ = {'spleen':1, 'left kidney'}
#     global_label = list(range(1, 40))  # list(np.unique(background_3d)[1:])
#     for i in range(2, rows * cols):
#         ind = start_with + i * show_every
#         # alpha是不透明度，值越小，越透明。
#         ##添加上窗宽窗位，就能跟一般医学图像那样显示CT了
#         img = adjust_ww_wl(img_roi[ind, :, :], ww=1500, wl=-500)
#         label = label_roi[ind, :, :]
#         label_type = np.unique(label)[1:]
#         # print(label_type)
#         ax[int(i / cols), int(i % cols)].set_title('slice%d' % ind)
#         ax[int(i / cols), int(i % cols)].imshow(img, cmap=plt.get_cmap('gray'))
#         color_list = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)
#         for n, j in enumerate(label_type):
#             mask_j = np.array(label == j, dtype=np.uint8)
#             points = np.concatenate([point for point in find_contours(mask_j, 0, 1)], axis=0)

#             ax[int(i / cols), int(i % cols)].scatter(points[:, 0], points[:, 1], s=2,
#                                                      color=color_list[global_label.index(j+1)],
#                                                      marker='.')  # alpha=0.8
#         # ax[int(i / cols), int(i % cols)].imshow(background_3d[ind, :, :], cmap=plt.get_cmap('tab20'), alpha = 0.5)
#         ax[int(i / cols), int(i % cols)].axis('off')
#     fig.suptitle(pat_id + str(exist_label))  # 'Patient_9mm'+ pat_id)
#     plt.show()
#     if bool(pat_id) and bool(save_path):
#         fig.savefig(os.path.join(save_path, pat_id))
#         plt.close()