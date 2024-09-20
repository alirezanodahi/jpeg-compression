from PIL import Image
import numpy as np
from scipy import fft



file_name = "testpic.jpg"


def convert_to_matrix(file_name):
    # Open image
    original_image = Image.open("./" + file_name)

    # Convert the image to a numpy array
    image_array = original_image.getdata()

    # Write image pixel into file
    with open("./pixels_data_rgb.txt", "w") as pixels_file:
        lines = 0
        for tup in image_array:
            if lines == 10:
                pixels_file.write("\n")
                lines = 0
            else:
                pixels_file.write(str(tup))
                lines += 1

    return image_array


original_img_pixels = convert_to_matrix(file_name)

# Create Y channel
y_component = []
for tup in original_img_pixels:
    result = tup[0] * 0.299 + tup[1] * 0.587 + tup[2] * 0.114
    y_component.append(result)


# Create U channel
u_component = []
for tup, y in zip(original_img_pixels, y_component):
    result = 0.492111 * (tup[2] - y)
    u_component.append(result)

# Create V channel
v_component = []
for tup, y in zip(original_img_pixels, y_component):
    result = 0.877283 * (tup[0] - y)
    v_component.append(result)


# def save_yuv_images():
#     np_array = np.array(y_component, dtype=np.float32)
#     np_array = np_array.reshape((225, 400))
#     y_image = Image.fromarray(np_array)
#     y_image.show("ychannel")

#     np_array = np.array(u_component, dtype=np.float32)
#     np_array = np_array.reshape((225, 400))
#     u_image = Image.fromarray(np_array)
#     # u_image.save("./u_ " + file_name + ".jpg")
#     u_image.show("uchannel")

#     np_array = np.array(v_component, dtype=np.float32)
#     np_array = np_array.reshape((225, 400))
#     v_image = Image.fromarray(np_array)
#     # v_image.save("./v_ " + file_name + ".jpg")
#     v_image.show("vchannel")


def create_8x8_blocks(channel):
    total_img_pixels = len(channel)

    # add 0 pixles to image to make it 8*8
    for i in range(64 - (total_img_pixels % 64)):
        channel.append(0)

    # Create 8*8 blocks and minus 128
    result_8x8_blocks = []
    row = []
    inner = []
    row_num = 0
    col_num = 0
    for pixel in channel:
        if col_num == 8:
            result_8x8_blocks.append(inner)
            inner = []
            col_num = 0
        else:
            if row_num == 8:
                inner.append(row)
                row = []
                row_num = 0
                col_num += 1
            else:
                row.append(pixel - 128)
                row_num += 1

    return result_8x8_blocks


# Create Cb channel
cb_component = []
for tup, y in zip(original_img_pixels, y_component):
    result = ((tup[2] - y) / 1.772) + 0.5
    cb_component.append(result)

# Create Cr channel
cr_component = []
for tup, y in zip(original_img_pixels, y_component):
    result = ((tup[0] - y) / 1.402) + 0.5
    cr_component.append(result)


y_8x8_blocks = create_8x8_blocks(y_component)
cb_8x8_blocks = create_8x8_blocks(cb_component)
cr_8x8_blocks = create_8x8_blocks(cr_component)


# Chroma subsampling 4:2:0
def chroma_sampling(matrix):
    # matrix is a numpy 2d array and 8*8 block
    result = []
    # temp = matrix.copy() # 4:2:0
    for block in matrix:
        temp = np.array(block)
        temp[1::2, :] = temp[::2, :]
        # Vertically, every second element equals to element above itself.
        temp[:, 1::2] = temp[:, ::2]
        # Horizontally, every second element equals to the element on its left side.
        result.append(temp.tolist())

    return result


cb_chroma_result = chroma_sampling(cb_8x8_blocks)
cr_chroma_result = chroma_sampling(cr_8x8_blocks)

# Using scipy.fft.dct() method
y_dct_result = fft.dct(y_8x8_blocks)
cb_dct_result = fft.dct(cb_chroma_result)
cr_dct_result = fft.dct(cr_chroma_result)


lumin_quantization_table = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]

chrom_quantization_table = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
]

# quantized value = dct_values // table
y_quantized = []
for outer in y_dct_result:
    matrix_8x8 = []
    for inner, qrow in zip(outer, lumin_quantization_table):
        vector_1x8 = []
        for val, q in zip(inner, qrow):
            vector_1x8.append(int(val // q))
        matrix_8x8.append(vector_1x8)
    y_quantized.append(matrix_8x8)


def chroma_quantization(channel):
    quantization_result = []
    for outer in channel:
        matrix_8x8 = []
        for inner, qrow in zip(outer, chrom_quantization_table):
            vector_1x8 = []
            for val, q in zip(inner, qrow):
                vector_1x8.append(int(val // q))
            matrix_8x8.append(vector_1x8)
        quantization_result.append(matrix_8x8)

    return quantization_result


cb_quantized = chroma_quantization(cb_dct_result)
cr_quantized = chroma_quantization(cr_dct_result)


def zigzag_traverse(matrix):
    zz_result = []
    for m in matrix:
        temp = np.array(m)
        zz_vector = np.concatenate(
            [
                np.diagonal(temp[::-1, :], i)[:: (2 * (i % 2) - 1)]
                for i in range(1 - temp.shape[0], temp.shape[0])
            ]
        )
        zz_result.append(list(zz_vector))
    return zz_result


zigzag_y = zigzag_traverse(y_quantized)
zigzag_cb = zigzag_traverse(cb_quantized)
zigzag_cr = zigzag_traverse(cr_quantized)


# DPCM on DC component
def run_dpcm(zz_matrix):
    dc_coefficient = []
    for vector in range(len(zz_matrix)):
        result = []
        if vector == 0:
            result = zz_matrix[vector].copy()
            dc_coefficient.append(result)
            continue
        for i in range(len(zz_matrix[vector])):
            result.append(zz_matrix[vector][i] - zz_matrix[vector - 1][i])
        dc_coefficient.append(result)
    return dc_coefficient


y_dc_coefficient = run_dpcm(zigzag_y)
cb_dc_coefficient = run_dpcm(zigzag_cb)
cr_dc_coefficient = run_dpcm(zigzag_cr)


# RLC on AC component
def run_rlc(matrix):
    count = 0
    rlc_result = []
    for vector in matrix:
        temp = []
        for i in range(len(vector) - 1):
            if i == 0:
                first = vector[i]
                temp.append(first)
            if vector[i + 1] == 0:
                count += 1
                continue
            else:
                tup = (count, vector[i + 1])
                count = 0
            temp.append(tup)
        rlc_result.append(temp)
    return rlc_result


y_rlc_coefficient = run_rlc(zigzag_y)
cb_rlc_coefficient = run_rlc(zigzag_cb)
cr_rlc_coefficient = run_rlc(zigzag_cr)


def write_result(filename, channel):
    with open(filename, "w") as file:
        for matrix in channel:
            lines = 0
            for comp in matrix:
                if lines == 10:
                    file.write("\n")
                    lines = 0
                else:
                    file.write(str(comp) + ", ")
                    lines += 1
            file.write("\n")


write_result("y_dc.txt", y_dc_coefficient)
write_result("cb_dc.txt", cb_dc_coefficient)
write_result("cr_dc.txt", cr_dc_coefficient)


write_result("y_rlc.txt", y_rlc_coefficient)
write_result("cb_rlc.txt", cb_rlc_coefficient)
write_result("cr_rlc.txt", cr_rlc_coefficient)
