from nilearn import plotting, image
import argparse


def show_image(image_path, cut_coords):
    img = image.load_img(image_path)
    plotting.plot_img(
        img, cut_coords=cut_coords, display_mode="ortho", title="NIFTI Image"
    )
    plotting.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a .nii.gz image.")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the .nii.gz image file."
    )
    parser.add_argument(
        "--cut_coords",
        type=float,
        nargs=3,
        required=True,
        help="The x, y, and z coordinates of the slice to display.",
    )
    args = parser.parse_args()

    # Plot the image using nilearn's plotting function
    show_image(args.image_path, args.cut_coords)


if __name__ == "__main__":
    main()
