# -*- coding: utf-8 -*-

"""
Trabalho Prático 1 - Multimédia

Alunos:
Joana Simões - 2019217013
Samuel Carinhas - 2019217199
Sofia Alves - 2019227240

"""

from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colors as clr
import numpy as np
import copy
import cv2
import scipy.fftpack as fft
import math as m

def read_image(image_name):
    """
    Reads an image from a file and returns a numpy array of its pixels.
    
    :param image_name: the name of the image to read
    :return: The image as a numpy array
    """
    image = np.array(plt.imread(image_name))
    return image

def create_colormap(color_list, name='cmap'):
    """
    Create a colormap from a list of colors
    
    :param color_list: a list of RGB values
    :param name: The name of the colormap, defaults to cmap (optional)
    :return: A colormap object.
    """
    return clr.LinearSegmentedColormap.from_list(name, color_list, N=256)

graymap = create_colormap(['black', 'white'], 'blackwhite')

def plot_image(image, colormap=graymap, title=""):
    """
    Plot an image using matplotlib
    
    :param image: The image to plot
    :param colormap: The colormap to use for the plot
    :param title: The title of the plot
    """
    plt.figure() 
    plt.title(title)
    plt.imshow(image, colormap)
    #plt.axis('off')
    plt.show()

def plot_compared_images(image1, image2, title1, title2):
    """
    Plot two images side by side
    
    :param image1: The first image to compare
    :param image2: The image to be compared to
    :param title1: The title of the first image
    :param title2: The title of the second image
    """
    fig = plt.figure(figsize=(10, 7)) 
    ax1 = fig.add_subplot(121)
    ax1.set_title(title1)
    ax1.imshow(image1, graymap)

    ax2 = fig.add_subplot(122)
    ax2.set_title(title2)
    ax2.imshow(image2, graymap)
    plt.show()

def get_image_rgb(image):
    """
    Given an image, return a tuple of three numpy arrays, one for each of the red, green, and blue
    channels
    
    :param image: the image to be converted to a numpy array
    :return: A tuple of three numpy arrays, each of which is an image channel.
    """
    return np.array((image[:, :, 0], image[:, :, 1], image[:, :, 2]))

def get_image_from_channels(channels):
    """
    Given a list of channels,
    return a 3-channel image
    
    :param channels: A list of the channels to use
    :return: a numpy array of shape (lines, columns, 3)
    """
    lines, columns = channels[0].shape
    img = np.zeros((lines, columns, 3), dtype=np.uint8)
    img[:, :, 0] = channels[0]
    img[:, :, 1] = channels[1]
    img[:, :, 2] = channels[2]
    return img

def add_padding(image, padding=16):
    """
    Given an image, it adds padding to the image so that the image is a multiple of a given number
    
    :param image: the image to be processed
    :param padding: , defaults to 16 (optional)
    :return: The method returns the image with padding
    """
    rows, columns, _ = image.shape
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    # add rows
    if rows % padding != 0:
        rows_to_add = padding - rows % padding

        aux_red = np.tile(red[-1, :], (rows_to_add, 1))
        aux_green = np.tile(green[-1, :], (rows_to_add, 1))
        aux_blue = np.tile(blue[-1, :], (rows_to_add, 1))

        red = np.vstack([red, aux_red])
        green = np.vstack([green, aux_green])
        blue = np.vstack([blue, aux_blue])
    
    # add columns
    if columns % padding != 0:
        columns_to_add = padding - columns % padding

        aux_red = np.tile(red[:, -1], (columns_to_add, 1))
        aux_green = np.tile(green[:, -1], (columns_to_add, 1))
        aux_blue = np.tile(blue[:, -1], (columns_to_add, 1))

        red = np.hstack([red, aux_red.T])
        green = np.hstack([green, aux_green.T])
        blue = np.hstack([blue, aux_blue.T])
    
    return get_image_from_channels((red, green, blue))

def revert_padding(image, original_rows, original_columns):
    """
    Given an image, revert the padding that was applied to the image
    
    :param image: The image to be cropped
    :param original_rows: the original number of rows in the image
    :param original_columns: The original width of the image
    :return: The image with the padding removed.
    """
    if(len(image.shape) < 3):
        return image[:original_rows, :original_columns]
    
    rows, columns, _ = image.shape
    
    if rows < original_rows or columns < original_columns:
        return image
    
    return image[:original_rows, :original_columns, :]

def convert_rgb_to_ycbcr(image):
    """
    Convert an RGB image to YCbCr
    
    :param image: The image to be converted to YCbCr
    :return: a numpy array of the same size as the input image, but with the YCbCr colorspace.
    """
    ycbcr_matrix = np.array([
                    [0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]])
                    
    aux = image.dot(ycbcr_matrix.T)
    aux[:, :, 1:3] += 128
    aux[aux > 255] = 255
    aux[aux < 0] = 0
    aux = aux.round()
    return np.uint8(aux)

def convert_ycbcr_to_rgb(image):
    """
    Convert an image from YCbCr to RGB
    
    :param image: the image to be converted
    :return: the image converted from YCbCr to RGB.
    """
    image = image.astype(np.float32)
    ycbcr_matrix = np.array([
                    [0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]])
                    
    inverse = np.linalg.inv(ycbcr_matrix.T)
    aux = np.copy(image)
    aux[:, :, 1:3] -= 128
    aux = aux.dot(inverse)
    aux[aux > 255] = 255
    aux[aux < 0] = 0
    aux = aux.round()
    return np.uint8(aux)

def downsampling(image, ratio, interpolation=False):
    """
    Given an image, it returns the downsampled version of the image
    
    :param image: The image to be downsampled
    :param ratio: The downsampling ratio
    :param interpolation: If False, use a faster algorithm, otherwise a slower but better one, defaults
    to False (optional)
    :return: a tuple of three images. The first one is the red channel, the second one is the green
    channel, and the third one is the blue channel.
    """
    
    ratios = {
        (4, 4, 4): (1, 1),
        (4, 4, 0): (1, 0.5),
        (4, 2, 2): (0.5, 1),
        (4, 2, 0): (0.5, 0.5),
        (4, 1, 1): (0.25, 1),
        (4, 1, 0): (0.25, 0.25)
    }

    scale_x, scale_y = ratios[ratio]

    if scale_x == 1 and scale_y == 1:
        return (image[:, :, 0], image[:, :, 1], image[:, :, 2])
    
    step_x = int(1//scale_x)
    step_y = int(1//scale_y)
    
    if interpolation:
        return (image[:, :, 0],
                cv2.resize(image[:, :, 1], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR),
                cv2.resize(image[:, :, 2], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR))
    else:
        return (image[:, :, 0], image[::step_y, ::step_x, 1], image[::step_y, ::step_x, 2])

def upsampling(y, cb, cr, ratio, interpolation=False):
    """
    Given a image, it will upsample the  channels by the given ratio and return the
    upsampled image
    
    :param y: The luma component of the image
    :param cb: Chroma Blue (U)
    :param cr: Chrominance component of the image
    :param ratio: The ratio of the input image to the output image
    :param interpolation: If True, uses bilinear interpolation for upsampling. Otherwise, uses nearest
    neighbor, defaults to False (optional)
    :return: the upsampled Y, Cb and Cr channels.
    """
    ratios = {
        (4, 4, 4): (1, 1),
        (4, 4, 0): (1, 0.5),
        (4, 2, 2): (0.5, 1),
        (4, 2, 0): (0.5, 0.5),
        (4, 1, 1): (0.25, 1),
        (4, 1, 0): (0.25, 0.25)
    }

    scale_x, scale_y = ratios[ratio]

    if scale_x == 1 and scale_y == 1:
        return (y, cb, cr)
    
    step_x = int(1//scale_x)
    step_y = int(1//scale_y)

    if interpolation:
        return (y,
            cv2.resize(cb, None, fx=step_x, fy=step_y, interpolation=cv2.INTER_LINEAR),
            cv2.resize(cr, None, fx=step_x, fy=step_y, interpolation=cv2.INTER_LINEAR))
    else:
        upsampled_cb = np.repeat(cb, step_x, axis=1)
        upsampled_cb = np.repeat(upsampled_cb, step_y, axis=0)

        upsampled_cr = np.repeat(cr, step_x, axis=1)
        upsampled_cr = np.repeat(upsampled_cr, step_y, axis=0)

        return (y, upsampled_cb, upsampled_cr)

def get_dct(channel):
    """
    Given a channel, return the DCT of the channel
    
    :param channel: The channel to be transformed
    :return: The dct of the channel.
    """
    return fft.dct(fft.dct(channel, norm="ortho").T, norm="ortho").T

def get_inverse_dct(channel):
    """
    Given a channel, return the inverse discrete cosine transform of that channel
    
    :param channel: The channel to be processed
    :return: The inverse dct of the channel.
    """
    return fft.idct(fft.idct(channel, norm="ortho").T, norm="ortho").T

def dct_block(channel, bs):
    """
    This function takes a channel and a block size and returns a dct of the channel in blocks
    
    :param channel: the channel of the image we want to compress
    :param bs: block size
    :return: The DCT coefficients of the image.
    """
    size = channel.shape
    dct = np.zeros(size)
    for i in np.r_[:size[0]:bs]:
        for j in np.r_[:size[1]:bs]:
            dct[i:(i+bs),j:(j+bs)] = get_dct(channel[i:(i+bs),j:(j+bs)])
    return dct

def idct_block(channel, bs):
    """
    This function performs the inverse discrete cosine transform on a block of the image
    
    :param channel: the channel of the image
    :param bs: block size
    :return: The inverse discrete cosine transform of the block.
    """
    size = channel.shape
    idct = np.zeros(size)
    for i in np.r_[:size[0]:bs]:
        for j in np.r_[:size[1]:bs]:
            idct[i:(i+bs),j:(j+bs)] = get_inverse_dct(channel[i:(i+bs),j:(j+bs)])
    idct[idct < 0] = 0
    idct[idct > 255] = 255
    return idct

def apply_quantization_block(channel, factor):
    """
    Given a channel, apply quantization by dividing each 8x8 block by a factor and rounding the result
    
    :param channel: the channel to be quantized
    :param factor: the quantization factor
    :return: The quantized image.
    """
    size = channel.shape
    quant = np.zeros(size, dtype=np.float32)
    for i in np.r_[:size[0]:8]:
        for j in np.r_[:size[1]:8]:
            quant[i:(i+8),j:(j+8)] = np.round(channel[i:(i+8),j:(j+8)] / factor)
    return quant

def apply_quantization_block_inverse(channel, factor):
    """
    Given a channel of the image, apply the inverse quantization block
    
    :param channel: the channel to be quantized
    :param factor: The quantization factor
    :return: The inverse quantization of the channel
    """
    size = channel.shape
    inverse_quant = np.zeros(size, dtype=np.float32)
    for i in np.r_[:size[0]:8]:
        for j in np.r_[:size[1]:8]:
            inverse_quant[i:(i+8),j:(j+8)] = channel[i:(i+8),j:(j+8)] * factor
    return inverse_quant

def calculate_quantization_factor(quality):
    """
    Given a quality factor, the function returns the quantization matrices for Y and CbCr components
    
    :param quality: The image quality, on a scale from 1 (worst) to 95 (best)
    :return: a tuple of two matrices.
    """
    if quality > 100:
        quality = 100
    if quality < 0:
        quality = 1
    qy = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
               [12, 12, 14, 19,  26,  58,  60,  55],
               [14, 13, 16, 24,  40,  57,  69,  56],
               [14, 17, 22, 29,  51,  87,  80,  62],
               [18, 22, 37, 56,  68, 109, 103,  77],
               [24, 35, 55, 64,  81, 104, 113,  92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103,  99]])
    qc = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])
    q_ones = np.ones((8, 8))
    scaling_factor = 0
    if quality >= 50:
        scaling_factor = (100 - quality) / 50
    else:
        scaling_factor = 50 / quality
    
    qy_factor = q_ones
    qc_factor = q_ones

    if scaling_factor != 0:
        qy_factor = np.round(qy * scaling_factor)
        qc_factor = np.round(qc * scaling_factor)
    
    qy_factor[qy_factor > 255] = 255
    qc_factor[qc_factor > 255] = 255
    qy_factor[qy_factor < 1] = 1
    qc_factor[qc_factor < 1] = 1
    
    return (qy_factor, qc_factor)

def quantization(y, cb, cr, quality=75):
    """
    Given the quantization factor, apply the quantization to the given channels
    
    :param y: The y channel of the image
    :param cb: The cb channel of the image
    :param cr: The cr channel of the image
    :param quality: a value between 1 and 100, defaults to 75 (optional)
    :return: The quantized y, cb, and cr values.
    """
    qy_factor, qc_factor = calculate_quantization_factor(quality)
    return (apply_quantization_block(y, qy_factor), apply_quantization_block(cb, qc_factor), apply_quantization_block(cr, qc_factor))

def inverse_quantization(y, cb, cr, quality=75):
    """
    Given the quantization factor, apply the inverse quantization to the given channels
    
    :param y: The y channel of the image
    :param cb: The cb channel of the image
    :param cr: The cr channel of the image
    :param quality: a value between 1 and 100, defaults to 75 (optional)
    :return: the inverse quantization of y, cb and cr channels
    """
    qy_factor, qc_factor = calculate_quantization_factor(quality)
    return (apply_quantization_block_inverse(y, qy_factor), apply_quantization_block_inverse(cb, qc_factor), apply_quantization_block_inverse(cr, qc_factor))

def dpcm(channel):
    """
    Given a channel, the function will return a channel with the same size, but with the DC coefficients
    encoded.
    
    :param channel: The channel to be processed
    :return: the encoded channel.
    """
    size = channel.shape
    dpcm_image = copy.deepcopy(channel.astype(np.float32))
    prev = channel[0, 0]
    for i in np.r_[:size[0]:8]:
        for j in np.r_[:size[1]:8]:
            if i == 0 and j == 0:
                continue
            dc = channel[i, j]
            dpcm_image[i, j] = dc - prev
            prev = dc

    return dpcm_image

def idpcm(channel):
    """
    Given a channel, the function returns the channel with the decoding DPCM values
    
    :param channel: The channel to be processed
    :return: the decoded channel.
    """
    size = channel.shape
    image = copy.deepcopy(channel.astype(dtype=np.float32))
    prev = channel[0, 0]
    for i in np.r_[:size[0]:8]:
        for j in np.r_[:size[1]:8]:
            if i == 0 and j == 0:
                continue
            image[i, j] = channel[i, j] + prev
            prev = image[i, j]

    return image

def encoder(original, ratio, interpolation, quality=75):
    """
    Given an image, the function will first add padding to the image, then convert the image to YCbCr, 
    downsample the image, apply DCT to each block, quantize the DCT coefficients, apply differential
    pulse-code modulation, and return the three DPCM coefficients.
    
    :param original: the original image
    :param ratio: The downsampling ratio
    :param interpolation: the interpolation method used for downsampling
    :param quality: the quality of the image, defaults to 75 (optional)
    :return: a tuple containing the dpcm coefficients of the Y, Cb and Cr channels and the original image size
    """
    #plot_image(original, title="Original image")
    shape = original[:, :, 0].shape
    image = add_padding(original)
    image = convert_rgb_to_ycbcr(image)
    y, cb, cr = downsampling(image, ratio, interpolation)
    y_d = dct_block(y, 8)
    cb_d = dct_block(cb, 8)
    cr_d = dct_block(cr, 8)

    y_quant, cb_quant, cr_quant = quantization(y_d, cb_d, cr_d, quality)
    y_dpcm = dpcm(y_quant)
    cb_dpcm = dpcm(cb_quant)
    cr_dpcm = dpcm(cr_quant)

    return (y_dpcm, cb_dpcm, cr_dpcm), shape

def decoder(channels, size, ratio, interpolation, quality=75):
    """
    Given the DPCM compressed channels, the quantization quality, and the interpolation and ratio, 
    we can reconstruct the image
    
    :param channels: The three channels of the image (y, cb, cr)
    :param size: The size of the original image
    :param ratio: The ratio used in to encode the channels
    :param interpolation: The interpolation value
    :param quality: The quality of the image,defaults to 75 (optional)
    :return: the reconstructed image.
    """
    y_idpcm = idpcm(channels[0])
    cb_idpcm = idpcm(channels[1])
    cr_idpcm = idpcm(channels[2])

    y_iquant, cb_iquant, cr_iquant = inverse_quantization(y_idpcm, cb_idpcm, cr_idpcm, quality)

    y_di = idct_block(y_iquant, 8)
    cb_di = idct_block(cb_iquant, 8)
    cr_di = idct_block(cr_iquant, 8)
    y, cb, cr = upsampling(y_di, cb_di, cr_di, ratio, interpolation)
    image = get_image_from_channels((y, cb, cr))
    image = convert_ycbcr_to_rgb(image)
    image = revert_padding(image, size[0], size[1])
    #plot_image(image, title="Reconstructed Image")

    return image

def mse_error(original, reconstructed):
    """
    Compute the mean squared error between the original and reconstructed images
    
    :param original: the original image
    :param reconstructed: the reconstructed image
    :return: The MSE error
    """
    size = original.shape
    mse = (1 / (size[0] * size[1])) * np.sum(np.power((original - reconstructed), 2))
    return mse

def rmse_error(mse):
    """
    Return the square root of the mean squared error
    
    :param mse: Mean squared error
    :return: rmse_error(mse)
    """
    return m.sqrt(mse)

def snr_error(original, mse):
    """
    Given an original image and an MSE value, calculate the SNR of the image
    
    :param original: the original image
    :param mse: the mean squared error between the original and the reconstructed image
    :return: The SNR value
    """
    size = original.shape
    p = (1/(size[0] * size[1])) * np.sum(np.power(original, 2))
    return 10 * m.log10(p / mse)

def psnr_error(original, mse):
    """
    Given an original image and an image with a certain error, 
    return the PSNR of the error
    
    :param original: the original image
    :param mse: The mean squared error between the two images
    :return: The PSNR value
    """
    return 10 * m.log10((np.max(original)**2) / mse)

def compare_results(image_name, quality, ratio=(4, 2, 0), interpolation=True):
    """
    Given an image, a quality, and a ratio, it encodes the image, decodes it, and compares the original
    image with the decoded image
    
    :param image_name: the name of the image to be used for the test
    :param quality: The quality of the jpeg image. This is a value between 1 and 100
    """
    original = read_image(f"./imagens/{image_name}.bmp")
    channels, shape = encoder(original, ratio, interpolation, quality)
    original_ycbcr = convert_rgb_to_ycbcr(original)
    image_r = decoder(channels, shape, ratio, interpolation, quality)
    img.imsave(f"{image_name}_{quality}.png", image_r)
    image_r_ycbcr = convert_rgb_to_ycbcr(image_r)
    diff_image = np.abs(original_ycbcr[:, : , 0].astype(np.int16) - image_r_ycbcr[:, :, 0].astype(np.int16)).astype(np.uint8)
    diff_image[0, 0] = 255

    plot_compared_images(image_r, diff_image, f"Reconstructed image - quality: {quality}", f"Difference image from quality {quality}")
    mse = mse_error(original.astype(np.float32), image_r.astype(np.float32))
    print("Diff Image: " + image_name + " Quality: " + str(quality))
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse_error(mse)))
    print("SNR: " + str(snr_error(original.astype(np.float32), mse)))
    print("PSNR: " + str(psnr_error(original.astype(np.float32), mse)))

def main():
    quality = 75
    image_name = "barn_mountains"
    ratio = (4, 2, 0)
    interpolation = True
    compare_results(image_name, quality, ratio, interpolation)

if __name__ == '__main__':
    main()
