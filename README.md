# JPEG Image Compression and DCT

This project implements a simplified JPEG image compression algorithm using **Discrete Cosine Transform (DCT)** and **Chroma Subsampling (4:2:0)**. The process includes converting the image to YCbCr color space, applying DCT, quantization, and performing compression techniques such as zigzag traversal, **DPCM (Differential Pulse Code Modulation)** on DC components, and **RLC (Run-Length Coding)** on AC components.

## Features
- **YCbCr Conversion**: Converts RGB image data to Y, Cb, and Cr channels.
- **Chroma Subsampling (4:2:0)**: Reduces the chroma resolution.
- **DCT (Discrete Cosine Transform)**: Applies DCT to the Y, Cb, and Cr channels.
- **Quantization**: Uses luminance and chrominance quantization tables.
- **Zigzag Traversal**: Reorders the quantized coefficients in a zigzag pattern.
- **DPCM (Differential Pulse Code Modulation)**: Compresses the DC components.
- **RLC (Run-Length Coding)**: Compresses the AC components.

## Installation

1. **Clone the repository**:
   - Clone the repository via a version control tool or download the project manually from GitHub.

2. **Install the required dependencies**:
   - The project uses `Pillow`, `numpy`, and `scipy` for image manipulation, array handling, and DCT computation respectively. Install the dependencies using a package manager like `pip`:
     ```python
     pip install Pillow numpy scipy
     ```

## Usage

1. **Convert the Image**: 
   The `convert_to_matrix` function reads the image file and writes the pixel data to `pixels_data_rgb.txt`.
   ```python
   original_img_pixels = convert_to_matrix(file_name)
   ```

2. **Y, Cb, Cr Channel Creation**: 
   The image is converted from the RGB color space to the YCbCr color space, which separates luminance (Y) from chrominance (Cb and Cr).

   - Y (Luminance) is created using a weighted sum of the RGB channels.
     ```python
     result = tup[0] * 0.299 + tup[1] * 0.587 + tup[2] * 0.114
     y_component.append(result)
     ```

   - Cb (Blue-Difference Chroma) and Cr (Red-Difference Chroma) are calculated as the differences between the RGB channels and Y.
     ```python
     result = ((tup[2] - y) / 1.772) + 0.5  # Cb
     cb_component.append(result)

     result = ((tup[0] - y) / 1.402) + 0.5  # Cr
     cr_component.append(result)
     ```

3. **Chroma Subsampling (4:2:0)**: 
   Chroma subsampling reduces the resolution of the Cb and Cr channels to save space. The 4:2:0 format keeps all Y data but downsamples Cb and Cr both horizontally and vertically.
   ```python
   cb_chroma_result = chroma_sampling(cb_8x8_blocks)
   cr_chroma_result = chroma_sampling(cr_8x8_blocks)
   ```

4. **DCT (Discrete Cosine Transform)**: 
   The DCT is applied to 8x8 blocks of the Y, Cb, and Cr channels to convert the pixel values into frequency coefficients.

   ```python
   y_dct_result = fft.dct(y_8x8_blocks)
   cb_dct_result = fft.dct(cb_chroma_result)
   cr_dct_result = fft.dct(cr_chroma_result)
   ```

5. **Quantization**: 
   The DCT coefficients are divided by the values in standard luminance and chrominance quantization tables to reduce the amount of data.

   - For Y (Luminance), a luminance quantization table is used:

     ```python
     for outer in y_dct_result:
         matrix_8x8 = []
         for inner, qrow in zip(outer, lumin_quantization_table):
             vector_1x8 = []
             for val, q in zip(inner, qrow):
                 vector_1x8.append(int(val // q))
             matrix_8x8.append(vector_1x8)
         y_quantized.append(matrix_8x8)
     ```

   - For Cb and Cr (Chrominance), a chrominance quantization table is used:

     ```python
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
     ```

6. **Zigzag Traversal**: 
   After quantization, the 8x8 blocks are converted into a 1D vector using a zigzag pattern, which prioritizes the non-zero values.
   
   ```python
   def zigzag_traverse(matrix):
       zz_result = []
       for m in matrix:
           temp = np.array(m)
           zz_vector = np.concatenate(
               [
                   np.diagonal(temp[::-1, :], i)[::(2 * (i % 2) - 1)]
                   for i in range(1 - temp.shape[0], temp.shape[0])
               ]
           )
           zz_result.append(list(zz_vector))
       return zz_result

   zigzag_y = zigzag_traverse(y_quantized)
   zigzag_cb = zigzag_traverse(cb_quantized)
   zigzag_cr = zigzag_traverse(cr_quantized)
     ```
7. **DPCM on DC Components**:
   The DC (Direct Current) components of each 8x8 block are compressed using Differential Pulse Code Modulation (DPCM). This technique records the difference between the DC coefficients of consecutive blocks, reducing redundancy and allowing for more efficient encoding.

   - The DC coefficients for the Y, Cb, and Cr channels are encoded as the difference between the DC value of the current block and the previous one.

   ```python
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
   ```
8. **Run-Length Coding (RLC) on AC Components**:
   After applying DPCM to the DC coefficients, the remaining AC (Alternating Current) coefficients of each block undergo Run-Length Coding (RLC). This method efficiently represents sequences of zeroes by recording only the number of zeroes followed by the next non-zero value, reducing the size of the encoded data.

   - The AC coefficients for the Y, Cb, and Cr channels are processed using RLC, which records each non-zero value and the count of preceding zeroes.

   ```python
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
   ```
9. **Save the Results**:
   The final coefficients after applying DPCM and RLC for the Y, Cb, and Cr channels are saved into separate text files. This allows for easy access and further processing of the encoded data.

   - The `write_result` function takes a filename and a channel (list of matrices) as input and writes the coefficients to the specified file. Each coefficient is formatted for readability.

   ```python
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
   ```
