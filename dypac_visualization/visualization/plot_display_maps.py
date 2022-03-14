import os
import matplotlib
import matplotlib.pyplot as plt
import nilearn
import nilearn.image
import nilearn.plotting
import nibabel
import pandas as pd


def _plot_dl_maps(img, cut_coords, annotated_name,
                  index, dimension, save_dir, resolution):
    display = nilearn.plotting.plot_stat_map(stat_map_img=img, cut_coords=cut_coords,
                                             colorbar=False, black_bg=False)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"{index}_{resolution}.jpg"),
                bbox_inches="tight", facecolor="w")
    plt.close()
    return display


def _save_results(annotated_names, maps_img, dimension, save_dir, resolution):
    maps_img = nibabel.load(maps_img)
    for i, img in enumerate(nilearn.image.iter_img(maps_img)):
        cut_coords = nilearn.plotting.find_xyz_cut_coords(img)
        if annotated_names is not None:
            annotated_name = annotated_names.iloc[i].Difumo_names
        else:
            annotated_name = None
        _plot_dl_maps(img, cut_coords, annotated_name, i, dimension, save_dir, resolution)
    return


def _get_values(maps_img, percent):
    """Get new intensities in each probabilistic map specified by
       value in percent. Useful for visualizing less non-overlapping
       between each probabilistic map.

    Parameters
    ----------
    maps_img : 4D Nifti image
        A 4D image which contains each dictionary/map in 3D.

    percent : float
        A value which is multiplied by true values in each probabilistic
        map.

    Returns
    -------
    values : list
        Values after multiplied by percent.
    """
    values = []
    for img in nilearn.image.iter_img(maps_img):
        values.append(img.get_data().max() * percent)
    return values


def plot_atlases_image(root_dir, percent=0.33, cmap=matplotlib.colors.ListedColormap('k', name='from_list', N=256)):

    components_to_display = [64, 128, 256, 512, 1024]
    resolutions_to_display = [2, 3]
    atlas_img_name = "tpl-MNI152NLin2009cAsym_res-0{res}_atlas-DiFuMo_desc-{dim}dimensions_probseg.nii.gz"
    atlas_label_name = "tpl-MNI152NLin2009cAsym_atlas-DiFuMo_desc-{dim}dimensions_probseg.tsv"

    for i, n_components in enumerate(components_to_display):
        for res in resolutions_to_display:
            maps_img = os.path.join(root_dir, "data", "processed", "difumo_atlases",
                                    "tpl-MNI152NLin2009cAsym", atlas_img_name.format(dim=n_components, res=res))
            label_path = os.path.join(root_dir, "data", "processed", "difumo_atlases",
                                    "tpl-MNI152NLin2009cAsym", atlas_label_name.format(dim=n_components, res=res))
            if percent is not None:
                threshold = _get_values(maps_img, percent)
            else:
                threshold = None
            display = nilearn.plotting.plot_prob_atlas(maps_img,
                                                       threshold=threshold, dim=0.1,
                                                       draw_cross=False,
                                                       cmap=cmap, linewidths=1.5)
            save_dir = os.path.join(
                root_dir, "reports", "imgs", "display_maps", str(n_components), str(res))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f"{n_components}_{res}.jpg"),
                        bbox_inches='tight')

            annotated_names = pd.read_csv(label_path, delimiter="\t")
            save_dir = os.path.join(
                root_dir, "reports", "imgs", "component_maps", str(n_components), str(res))
            _save_results(annotated_names, maps_img, n_components, save_dir, res)


def main():

    root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    percent = 0.33
    cmap = matplotlib.colors.ListedColormap('k', name='from_list', N=256)

    plot_atlases_image(
        root_dir=root_dir, percent=percent, cmap=cmap)


if __name__ == '__main__':
    main()
