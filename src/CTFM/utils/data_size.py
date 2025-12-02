import numpy as np

def tensor_memory_report(num_images, channels, height, width, dtype='float32'):
    """
    Compute memory usage for storing image tensors.

    Args:
        num_images (int): Number of images (N)
        channels (int): Number of channels (C)
        height (int): Image height (H)
        width (int): Image width (W)
        dtype (str): Data type (default: 'float32')
                     Examples: 'float32', 'float16', 'uint8'

    Prints:
        Total elements, bytes, MB, MiB, GB, GiB
    """
    # Compute basic quantities
    bytes_per_elem = np.dtype(dtype).itemsize
    num_elements = num_images * channels * height * width
    total_bytes = num_elements * bytes_per_elem

    # Conversions
    MB = total_bytes / (1000**2)
    MiB = total_bytes / (1024**2)
    GB = total_bytes / (1000**3)
    GiB = total_bytes / (1024**3)

    # Print summary
    print(f"Tensor shape: ({num_images}, {channels}, {height}, {width}), dtype={dtype}")
    print(f"Total elements: {num_elements:,}")
    print(f"Bytes per element: {bytes_per_elem}")
    print(f"Total bytes: {total_bytes:,}")
    print(f"≈ {MB:,.2f} MB (decimal)   |  {MiB:,.2f} MiB (binary)")
    print(f"≈ {GB:,.3f} GB (decimal)   |  {GiB:,.3f} GiB (binary)")

if __name__ == "__main__":
    exams = 1954
    images_per_exam = 200
    num_images = exams * images_per_exam
    channels = 4
    height = 64
    width = 64

    tensor_memory_report(num_images, channels, height, width)

