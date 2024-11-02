import numpy as np
import matplotlib.pyplot as plt
from fDBSCAN.sim_pdf_abnormal import SimPDFAbnormal

# Set random seed for data
random_seed = 24
np.random.seed(random_seed)


if __name__ == '__main__':

    # Declare param input
    start = -50
    stop = 50
    step = 0.2

    # Create range data
    data = np.arange(start, stop + step, step)
    
    # Generate probability density functions (PDFs) for abnormal functions
    # Generate random means for the PDFs
    mu1 = np.random.normal(10, 5, 20)
    mu2 = np.random.normal(-10, 2, 5)

    # Generate the PDFs and true labels
    f_x, f_y = SimPDFAbnormal(
        [mu1, mu2],
        [6, 9],
        data
    )

    # Plot the original PDFs in gray color
    if False:
        plt.figure(figsize=(12, 6))
        for i in range(f_x.shape[1]):
            plt.plot(data, f_x[:, i], color=[0.8, 0.8, 0.8])
        plt.title('Original PDFs')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.show()