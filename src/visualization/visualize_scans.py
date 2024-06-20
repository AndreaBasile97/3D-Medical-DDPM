from nilearn import plotting, image


def show_MRI(image_path):
    img = image.load_img(image_path)
    plotting.plot_img(img)
    plotting.show()


if __name__ == "__main__":
    image_path = "data/raw/HC_T1/NIFD_1_S_0071_MR_T1_mprage__br_raw_20120819160257384_19_S161460_I324709.nii"
    show_MRI(image_path)
