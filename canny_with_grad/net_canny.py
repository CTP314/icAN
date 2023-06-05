# canny edge detection modifed from https://github.com/DCurro/CannyEdgePytorch
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian
import matplotlib.pyplot as plt

eps = 1e-5

class Net(nn.Module):
    def __init__(self, threshold=3.0, use_cuda=False, device=None):
        super(Net, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 3
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), device=device)
        self.gaussian_filter_horizontal1.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal1.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_horizontal2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), device=device)
        self.gaussian_filter_horizontal2.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal2.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_horizontal3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), device=device)
        self.gaussian_filter_horizontal3.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal3.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        
        
        self.gaussian_filter_vertical1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), device=device)
        self.gaussian_filter_vertical1.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical1.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), device=device)
        self.gaussian_filter_vertical2.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical2.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), device=device)
        self.gaussian_filter_vertical3.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical3.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, device=device)
        self.sobel_filter_horizontal1.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal1.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_horizontal2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, device=device)
        self.sobel_filter_horizontal2.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal2.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_horizontal3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, device=device)
        self.sobel_filter_horizontal3.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal3.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        
        self.sobel_filter_vertical1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, device=device)
        self.sobel_filter_vertical1.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical1.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, device=device)
        self.sobel_filter_vertical2.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical2.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2, device=device)
        self.sobel_filter_vertical3.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical3.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2, device=device)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))
        
        self.device = device

    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal1(img_r)
        blurred_img_r = self.gaussian_filter_vertical1(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal2(img_g)
        blurred_img_g = self.gaussian_filter_vertical2(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal3(img_b)
        blurred_img_b = self.gaussian_filter_vertical3(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal1(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical1(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal2(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical2(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal3(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical3(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2 + eps) 
        grad_mag = grad_mag + torch.sqrt(grad_x_g**2 + grad_y_g**2 + eps)
        grad_mag = grad_mag + torch.sqrt(grad_x_b**2 + grad_y_b**2 + eps)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)
        # print(all_filtered.shape)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = pixel_range.to(self.device)

        # print(inidices_positive.data.shape)
        bsz = inidices_positive.data.shape[0]
        indices = (inidices_positive.view(bsz, -1).data * pixel_count + pixel_range).squeeze()
        addition = (torch.FloatTensor([range(bsz) for i in range(pixel_count)]).T * (8 * pixel_count)).to(self.device)
        # print(addition.shape)
        # print(indices.shape)
        # print(all_filtered.view(-1).shape)
        # print(all_filtered.view(-1)[indices.long()].shape)
        channel_select_filtered_positive = all_filtered.view(-1)[(indices+addition).long()].view(bsz, 1, height,width)
        # print(f'shape = {channel_select_filtered_positive.shape}')
        
        # image = (channel_select_filtered_positive.data.cpu().numpy()[2, 0]).astype(float)
        # plt.imshow(image)
        # plt.savefig("test11.png")

        indices = (inidices_negative.view(bsz, -1).data * pixel_count + pixel_range).squeeze()
        # print(indices.max())
        channel_select_filtered_negative = all_filtered.view(-1)[(indices+addition).long()].view(bsz, 1, height,width)
        # print(channel_select_filtered_negative.shape)
        
        # image = (channel_select_filtered_negative.data.cpu().numpy()[0, 0]).astype(float)
        # plt.imshow(image)
        # plt.savefig("test10.png")

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        # print(is_max.shape)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


if __name__ == '__main__':
    Net()