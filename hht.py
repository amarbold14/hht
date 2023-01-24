from shared.MyTrainer import *
from shared.MyWriter import MyWriter
from shared.data_config import *
from shared.MyEvaluator import *
from shared.utility import SharedVariable
import matplotlib
import emd
import io
from PIL import Image
import torchvision.transforms as transforms


def hht_transform(x):
    # Use buffer for memory management
    buffer = io.BytesIO()
    myFig = matplotlib.pyplot.figure()

    # Sample rate ~70hz taken from RosA
    sample_rate = 70

    # Define freq edges
    freq_edges, freq_centres = emd.spectra.define_hist_bins(1, 50, 20, 'log')

    # EMD
    imf = emd.sift.mask_sift(x, max_imfs=5)

    # Instantanious phase, freq, and amplitude
    IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')

    f, hht = emd.spectra.hilberthuang(IF[:,2], IA[:,2], freq_edges, mode='amplitude', sum_time=False)

    # Plot the spectrum
    time_centres = np.arange(2000)-0.5
    emd.plotting.plot_hilberthuang(hht, time_centres, freq_centres, cmap='viridis', log_y=False, fig=myFig).figure.savefig(buffer)

    buffer.seek(0)
    im = Image.open(buffer)
    matplotlib.pyplot.close(myFig)

    return buffer, im


def get_image(data):
    data = data.transpose()
    x,y,z = data[0,:], data[1,:], data[2,:]

    # Process x,y,z acceleration feature extraction
    buf_x, im_x = hht_transform(x)
    buf_y, im_y = hht_transform(y)
    buf_z, im_z = hht_transform(z)

    # Crop unwanted parts of the plot
    im_x_cropped = im_x.crop((80, 58, 640, 428))
    im_y_cropped = im_y.crop((80, 58, 640, 428))
    im_z_cropped = im_z.crop((80, 58, 640, 428))

    dst = Image.new('RGB', (im_x_cropped.width, im_x_cropped.height * 3))
    dst.paste(im_x_cropped, (0, 0))
    dst.paste(im_y_cropped, (0, im_x_cropped.height))
    dst.paste(im_z_cropped, (0, im_x_cropped.height * 2))

    # Close the buffers
    buf_x.close()
    buf_y.close()
    buf_z.close()
    out = dst.resize((224, 224))

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # out.show()
    out_tensor = transform(out)

    return out_tensor







