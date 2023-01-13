import json

import cv2
import torch
import numpy as np
import torch.nn as nn
import torchpwl
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
# from models.fastrcnn import FasterRCNN
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt


class PWLToneMapping(nn.Module):
    def __init__(self, n_breakpoints=30):
        super(PWLToneMapping, self).__init__()
        self.n_breakpoints = n_breakpoints
        self.b_left = 0
        self.b_right = 1
        self.interval = (self.b_right - self.b_left) / self.n_breakpoints

        # self.y_pos = []
        # for i in np.linspace(self.b_left, self.b_right, self.n_breakpoints + 1):
        #     self.y_pos.append(torch.nn.Parameter(torch.Tensor([i])))

        self.y_pos = torch.nn.Parameter(torch.linspace(self.b_left, self.b_right, self.n_breakpoints + 1))

    def forward(self, x):
        segment_idx = torch.floor((x - self.b_left) / self.interval)
        segment_idx = torch.clip(segment_idx, 0, self.n_breakpoints-1).type(torch.int64)  # Segment index of each pixel

        segment_xl = self.b_left + segment_idx * self.interval  # The left boundary of the x-axis of each segment
        segment_yl = self.y_pos[segment_idx]  # The left boundary of the y-axis of each segment
        segment_yr = self.y_pos[segment_idx+1]  # The right boundary of the y-axis of each segment
        segment_slope = (segment_yr - segment_yl) / self.interval  # Slope of each segment
        out = ((x - segment_xl) * segment_slope + segment_yl)
        out = out.clip(0, 1)
        return [out]


class GammaDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pwl_tm = PWLToneMapping()
        self.pwl_tm.train()

        det_model_weights_fp = '/Users/yjunj/Projs/barcode-detection/frcnn_resnet50_fpn.pth'
        # self.det = fasterrcnn_resnet50_fpn(num_classes=3)
        backbone = resnet50()
        backbone = _resnet_fpn_extractor(backbone, 0)
        self.det = FasterRCNN(backbone, num_classes=3, box_nms_thresh=0.1, box_score_thresh=0.3)
        self.det.load_state_dict(torch.load(det_model_weights_fp))
        self.det.eval()
        for param in self.det.parameters():
            param.requires_grad = False

    def forward(self, image, target=None):
        tm_image = self.pwl_tm(image)
        losses, detections = self.det(tm_image, target)
        return losses, detections, tm_image


def plot(n_breakpoints, y):
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('Input', fontsize=15)
    plt.ylabel('Output', fontsize=15)
    plt.title('PWLNNs ToneMapping', fontsize=20)

    x = np.linspace(0, 1, num=n_breakpoints+1)
    plt.plot(list(x), list(y))
    fig.canvas.draw()
    curve = np.array(fig.canvas.renderer.buffer_rgba())
    return curve[:, :, :-1]


def phase_json(json_fp):
    content = json.load(open(json_fp, 'r'))
    shapes = content['shapes']
    boxes = []
    labels = []
    for shape in shapes:
        x1, y1 = shape['points'][0]
        x2, y2 = shape['points'][1]
        label = int(shape['label'])
        boxes.append([x1, y1, x2, y2])
        labels.append(label)
    boxes = torch.as_tensor(boxes, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    target = {'boxes': boxes, 'labels': labels}
    return target


def main():
    video_writer = cv2.VideoWriter(
        filename="process-of-pwlnns.avi",
        fourcc=cv2.VideoWriter_fourcc('M','J','P','G'),
        fps=5,
        frameSize=(3360, 800)
    )
    model = GammaDet()
    optimizer = torch.optim.Adam(model.pwl_tm.parameters(), lr=5e-2)

    # img_path = '/Users/yjunj/Downloads/10bit.png'
    # target = phase_json('/Users/yjunj/Downloads/10bit.json')

    img_path = '/Users/yjunj/Downloads/raw10_3.png'
    target = phase_json('/Users/yjunj/Downloads/raw8_3.json')

    # img_path = '/Users/yjunj/Projs/exp-fusion/raw10_1.png'
    # target = phase_json('/Users/yjunj/Downloads/raw10_1.json')

    img = (cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 1023).astype(np.float32)
    image_tensor = F.to_tensor(img)
    raw_img = (np.stack((img, img, img), axis=2)*255).astype(np.uint8)
    for i in range(200):
        losses, detections, tm_img = model(image_tensor, [target])

        loss = sum(loss for loss in losses.values())

        optimizer.zero_grad()
        loss.backward()
        if i < 5:
            optimizer.param_groups[0]['params'][0].grad[20:] = 0
            optimizer.defaults['lr'] = 1e-4
        elif i < 10:
            optimizer.param_groups[0]['params'][0].grad[:-20] = 0
            optimizer.defaults['lr'] = 5e-5
        else:
            optimizer.defaults['lr'] = 1e-5
        optimizer.step()

        model.pwl_tm.y_pos.data = torch.clip(model.pwl_tm.y_pos, 0, 1)
        prev = model.pwl_tm.y_pos.data[0]
        for i in range(1, 31):
            cur = model.pwl_tm.y_pos.data[i]
            if cur < prev:
                model.pwl_tm.y_pos.data[i] = (cur+prev)/2
            prev = cur

        y_pos = [round(i, 2) for i in model.pwl_tm.y_pos.tolist()]
        print(f'loss = {loss}, '
              f'ys = {*y_pos,}')
        curve = plot(n_breakpoints=30, y=y_pos)

        boxes = detections[0]['boxes']
        vis_image = draw_bounding_boxes((tm_img[0] * 255).to(torch.uint8),
                                        boxes, colors='blue')
        vis_image = np.asarray(vis_image).transpose((1, 2, 0))

        vis_image = np.concatenate((raw_img, vis_image, curve), axis=1)
        cv2.putText(vis_image, 'Raw Image', (10, 60), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 0), thickness=2)
        cv2.rectangle(vis_image, (0, 0), (1280, 800), (255, 255, 255), 3)

        cv2.putText(vis_image, 'TM Image', (1290, 60), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 0), thickness=2)
        cv2.rectangle(vis_image, (1280, 0), (2560, 800), (255, 255, 255), 3)

        cv2.imshow("PWL-TM Visualization", vis_image)
        cv2.waitKey(10)
        video_writer.write(vis_image)

    # cv2.waitKey(0)
    video_writer.release()


if __name__ == '__main__':
    main()
