import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns

def save_phac_data_in_image_form(normalize = True):
    # Load presaved entities
    npz_loader = np.load("./entity/imgs_adjs_names.npz", allow_pickle=True)
    # Fetch images
    imgs = npz_loader["arr_0"]
    # Fetch adjectives
    adjs = npz_loader["arr_1"]
    # Fetch names
    names = npz_loader["arr_2"]


    for k in tqdm(range(imgs.shape[0]), colour = "red"):
        # Initialize figure
        plt.figure()
        # Fetch each image
        img = imgs[k, :, :]
        if normalize:
            # Max-min normalization
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # Draw heatmap of the image
        ax = sns.heatmap(img, cmap = "gist_rainbow")
        # Set title
        plt.title("{0}\n{1}".format(names[k], "-".join([adj.decode("utf-8") for adj in adjs[k]])))

        if normalize:
            # Save figure
            plt.savefig("./entity/imgs/normalized/{0}-{1}-normalized.png".format(str(k), names[k]))
        else:
            plt.savefig("./entity/imgs/{0}-{1}.png".format(str(k), names[k]))
        # Close figure not to bloat the memory
        plt.close()



if __name__ == "__main__":

    save_phac_data_in_image_form()