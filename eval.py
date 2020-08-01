import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.CE2P import Res_Deeplab
from dataset.datasets import LIPDataSet
from utils.utils import decode_parsing, inv_preprocess
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from copy import deepcopy
import scipy.misc

DATA_DIRECTORY = '/ssd1/liuting14/Dataset/LIP/'
DATA_LIST_PATH = './dataset/list/lip/valList.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = (473,473)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    return parser.parse_args()

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    interp_1 = torch.nn.Upsample(size=(384, 384), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, label_parsing, label_r0, label_r1, label_r2, label_r3, label_l0, label_l1, label_l2, label_l3, label_l4, label_l5, label_edge, meta = batch
            num_images = image.size(0)
            if index % 10 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs = model(image.cuda())
            if gpus > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                #gt = torch.from_numpy(parsing_anno)
                gt_parsing_colors = decode_parsing(label_parsing, 2, 20, False)
                gt_r0_colors = decode_parsing(label_r0, 2, 20, False)
                gt_r1_colors = decode_parsing(label_r1, 2, 20, False)
                gt_r2_colors = decode_parsing(label_r2, 2, 20, False)
                gt_r3_colors = decode_parsing(label_r3, 2, 20, False)
                #np.set_printoptions(threshold=np.inf)
                #print(label_l0.numpy())
                gt_l0_colors = decode_parsing(label_l0, 2, 20, False)
                gt_l1_colors = decode_parsing(label_l1, 2, 20, False)
                gt_l2_colors = decode_parsing(label_l2, 2, 20, False)
                gt_l3_colors = decode_parsing(label_l3, 2, 20, False)
                gt_l4_colors = decode_parsing(label_l4, 2, 20, False)
                gt_l5_colors = decode_parsing(label_l5, 2, 20, False)
                for i in range(2):
                    scipy.misc.toimage(gt_parsing_colors[i]).save("./pics/{}_{}_gt.png".format(index, i))
                    scipy.misc.toimage(gt_r0_colors[i]).save("./pics/{}_{}_gt_r0.png".format(index, i))
                    scipy.misc.toimage(gt_r1_colors[i]).save("./pics/{}_{}_gt_r1.png".format(index, i))
                    scipy.misc.toimage(gt_r2_colors[i]).save("./pics/{}_{}_gt_r2.png".format(index, i))
                    scipy.misc.toimage(gt_r3_colors[i]).save("./pics/{}_{}_gt_r3.png".format(index, i))
                    scipy.misc.toimage(gt_l0_colors[i]).save("./pics/{}_{}_gt_l0.png".format(index, i))
                    scipy.misc.toimage(gt_l1_colors[i]).save("./pics/{}_{}_gt_l1.png".format(index, i))
                    scipy.misc.toimage(gt_l2_colors[i]).save("./pics/{}_{}_gt_l2.png".format(index, i))
                    scipy.misc.toimage(gt_l3_colors[i]).save("./pics/{}_{}_gt_l3.png".format(index, i))
                    scipy.misc.toimage(gt_l4_colors[i]).save("./pics/{}_{}_gt_l4.png".format(index, i))
                    scipy.misc.toimage(gt_l5_colors[i]).save("./pics/{}_{}_gt_l5.png".format(index, i))

                parsing = outputs[0][0]
                tmp = interp_1(parsing)
                tmp = torch.argmax(tmp, dim=1, keepdim=False)
                ignore_index = label_parsing == 255
                tmp[ignore_index] = 0
                preds_colors = decode_parsing(tmp, 2, 20, False)
                pred_r0 = outputs[1][0]
                pred_r0 = interp_1(pred_r0)
                pred_r0_colors = decode_parsing(pred_r0, 2, 20, True)
                pred_r1 = outputs[1][1]
                pred_r1 = interp_1(pred_r1)
                pred_r1_colors = decode_parsing(pred_r1, 2, 20, True)
                pred_r2 = outputs[1][2]
                pred_r2 = interp_1(pred_r2)
                pred_r2_colors = decode_parsing(pred_r2, 2, 20, True)
                pred_r3 = outputs[1][3]
                pred_r3 = interp_1(pred_r3)
                pred_r3_colors = decode_parsing(pred_r3, 2, 20, True)
                pred_l0 = outputs[2][0]
                pred_l0 = interp_1(pred_l0)
                pred_l0_colors = decode_parsing(pred_l0, 2, 20, True)
                pred_l1 = outputs[2][1]
                pred_l1 = interp_1(pred_l1)
                pred_l1_colors = decode_parsing(pred_l1, 2, 20, True)
                pred_l2 = outputs[2][2]
                pred_l2 = interp_1(pred_l2)
                pred_l2_colors = decode_parsing(pred_l2, 2, 20, True)
                pred_l3 = outputs[2][3]
                pred_l3 = interp_1(pred_l3)
                pred_l3_colors = decode_parsing(pred_l3, 2, 20, True)
                pred_l4 = outputs[2][4]
                pred_l4 = interp_1(pred_l4)
                pred_l4_colors = decode_parsing(pred_l4, 2, 20, True)
                pred_l5 = outputs[2][5]
                pred_l5 = interp_1(pred_l5)
                pred_l5_colors = decode_parsing(pred_l5, 2, 20, True)
                for i in range(2):
                    scipy.misc.toimage(preds_colors[i]).save("./pics/{}_{}_pred.png".format(index, i))
                    scipy.misc.toimage(pred_r0_colors[i]).save("./pics/{}_{}_pred_r0.png".format(index, i))
                    scipy.misc.toimage(pred_r1_colors[i]).save("./pics/{}_{}_pred_r1.png".format(index, i))
                    scipy.misc.toimage(pred_r2_colors[i]).save("./pics/{}_{}_pred_r2.png".format(index, i))
                    scipy.misc.toimage(pred_r3_colors[i]).save("./pics/{}_{}_pred_r3.png".format(index, i))
                    scipy.misc.toimage(pred_l0_colors[i]).save("./pics/{}_{}_pred_l0.png".format(index, i))
                    scipy.misc.toimage(pred_l1_colors[i]).save("./pics/{}_{}_pred_l1.png".format(index, i))
                    scipy.misc.toimage(pred_l2_colors[i]).save("./pics/{}_{}_pred_l2.png".format(index, i))
                    scipy.misc.toimage(pred_l3_colors[i]).save("./pics/{}_{}_pred_l3.png".format(index, i))
                    scipy.misc.toimage(pred_l4_colors[i]).save("./pics/{}_{}_pred_l4.png".format(index, i))
                    scipy.misc.toimage(pred_l5_colors[i]).save("./pics/{}_{}_pred_l5.png".format(index, i))
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]


    return parsing_preds, scales, centers

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))

    input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = LIPDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    num_samples = len(lip_dataset)

    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

    restore_from = args.restore_from

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))

    mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)

    print(mIoU)

if __name__ == '__main__':
    main()
