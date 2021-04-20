import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns

def save_phac_data_in_image_form():
    # Load presaved entities
    npz_loader = np.load("./entity/imgs_imgs_normalized_adjs_names.npz", allow_pickle=True)
    # Fetch images
    imgs = npz_loader["arr_0"]
    # Fetch images
    imgs_normalized = npz_loader["arr_1"]
    # Fetch adjectives
    adjs = npz_loader["arr_2"]
    # Fetch names
    names = npz_loader["arr_3"]


    for k in tqdm(range(imgs.shape[0]), colour = "red"):
       
        # Fetch each image
        img = imgs[k, :, :]
        img_normalized = imgs_normalized[k, :, :]
        # Initialize figure

        for (im, fn) in zip([img, img_normalized], ["normal", "normalized"]):


            plt.figure()
            # Draw heatmap of the image
            ax = sns.heatmap(im, cmap = "gist_rainbow")
            # Set title
            plt.title("{0}\n{1}".format(names[k], "-".join(adjs[k])))
            plt.savefig("./entity/imgs/{2}/{0}-{1}.png".format(str(k), names[k], fn))
            # Close figure not to bloat the memory
            plt.close()



if __name__ == "__main__":

    save_phac_data_in_image_form()