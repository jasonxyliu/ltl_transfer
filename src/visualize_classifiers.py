import argparse
import os
import dill
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as plticker
from zero_shot_transfer import construct_initiation_set_classifiers

Image.MAX_IMAGE_PIXELS = None
LANDMARK_SZ = (5, 5)
ID2NAME = {
    "a": "wood",
    "b": "toolshed",
    "c": "workbench",
    "d": "grass",
    "e": "factory",
    "f": "iron",
    "g": "bridge",
    "h": "axe",
    "s": "shelter",  # rectangular region of multiple grid cells
    "X": "obstacle",
}


def run_visualizer(algo, ltl_id, classifier_dpath, map_fpath, vis_dpath, simple_vis=True):
    with open(os.path.join(classifier_dpath, "completed_ltls.pkl"), "rb") as file:
        completed_ltls = dill.load(file)["completed_ltl"]
    completed_ltls.sort()
    if ltl_id == -1:
        for ltl_id in completed_ltls:
            visualize_discrete_classifier(algo, ltl_id, classifier_dpath, map_fpath, vis_dpath, simple_vis)
    else:
        visualize_discrete_classifier(algo, ltl_id, classifier_dpath, map_fpath, vis_dpath, simple_vis)


def visualize_discrete_classifier(algo, ltl_id, classifier_dpath, map_fpath, vis_dpath, simple_vis):
    out_fpath = os.path.join(vis_dpath, "output.txt")
    with open(out_fpath, "a") as file:
        file.write("\nGenerating visualization for LTL: %d\n" % ltl_id)
    rollout_results_fpath = os.path.join(classifier_dpath, "rollout_results_parallel.pkl")
    if os.path.exists(rollout_results_fpath):
        with open(rollout_results_fpath, "rb") as rfile:
            policy2loc2edge2hits = dill.load(rfile)
        policy2edge2loc2prob = construct_initiation_set_classifiers(classifier_dpath)
        ltl = policy2loc2edge2hits["ltls"][ltl_id]
        edge2loc2prob = policy2edge2loc2prob[ltl]  # classifiers to visualize
    else:
        with open(out_fpath, "a") as file:
            file.write("FileNotFound: aggregated rollout results\n%s\nConstructing from single worker results\n" % rollout_results_fpath)
        with open(os.path.join(classifier_dpath, "completed_ltls.pkl"), "rb") as file:
            completed_ltls = dill.load(file)["completed_ltl"]
        completed_ltls.sort()
        # print("Completed LTLs: ", completed_ltls)
        if ltl_id not in completed_ltls:
            raise Exception("Requested LTL %d is not completed\nCompeted LTLs are %s" % (ltl_id, completed_ltls))
        edge2loc2prob = defaultdict(dict)
        nstates = 0
        for fname in os.listdir(classifier_dpath):
            if "ltl%d" % ltl_id in fname and fname.endswith('.pkl'):
                worker_results_fpath = os.path.join(classifier_dpath, fname)
                with open(worker_results_fpath, "rb") as file:
                    worker_results = dill.load(file)
                edge2hits = worker_results["edge2hits"]
                loc = worker_results["state"]
                # if loc != (3, 11):
                #     continue
                n_rollouts = worker_results["n_rollouts"] if "n_rollouts" in worker_results else 100
                if not edge2hits:
                    with open(out_fpath, "a") as file:
                        file.write("ATTENTION: LTL %d 0 hits from location: %s\n" % (ltl_id, str(loc)))
                for edge, hits in edge2hits.items():
                    # print(loc, edge, hits)
                    edge2loc2prob[edge][loc] = hits / n_rollouts
                nstates += 1
        # print(edge2loc2prob)
        with open(out_fpath, "a") as file:
            file.write("Rolled out LTL %d from %d locations\n" % (ltl_id, nstates))

    if simple_vis:
        simple_visualizer(vis_dpath, edge2loc2prob, ltl_id)
    else:
        landmark_dpath = os.path.join("../vis", "minecraft")
        decorated_map_fpath = os.path.join(vis_dpath, "map.png")
        if not os.path.exists(decorated_map_fpath):
            decorate_map(map_fpath, landmark_dpath, decorated_map_fpath)
        add_heat_map(decorated_map_fpath, vis_dpath, edge2loc2prob, ltl_id)


def simple_visualizer(vis_dpath, edge2loc2prob, ltl_id):
    """
    Construct a heatmap with a simple map where propositions are represented by letters not landmark images

    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    map_array = load_map(map_fpath)
    grid_dim = map_array.shape[0]  # assume nrows == ncols
    cmap = "Wistia"

    for edge, loc2prob in edge2loc2prob.items():
        classifier = np.zeros((grid_dim, grid_dim))
        for loc, prob in loc2prob.items():
            classifier[loc] = prob

        fig, ax = plt.subplots()
        im = ax.imshow(classifier, cmap=cmap)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap)
        cbar.ax.set_ylabel(ylabel="probability of success", rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(grid_dim), labels=list(range(grid_dim)))
        ax.set_yticks(np.arange(grid_dim), labels=list(range(grid_dim)))

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(classifier.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(classifier.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="k", alpha=0.5, linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Change the text's color depending on the data
        threshold = im.norm(np.max(classifier)) / 2.
        textcolors = ("black", "white")

        # Loop over data dimensions and create text annotations.
        for i in range(grid_dim):
            for j in range(grid_dim):
                text = ax.text(j, i, map_array[i, j], ha="center", va="center", fontweight='bold',
                               color=textcolors[int(im.norm(classifier[i, j]) > threshold)])

        # ax.set_title("ltl_%d_edge_%s_simple" % (ltl_id, edge))
        fig.tight_layout()
        vis_classifier_fpath = os.path.join(vis_dpath, "ltl_%d_edge_%s_simple.png" % (ltl_id, edge))
        plt.savefig(vis_classifier_fpath)


def decorate_map(map_fpath, landmark_dpath, save_fpath):
    """
    Read in map_x.txt, plot map figure with minecraft landmarks

    Plot a grid of images:
    https://kanoki.org/2021/05/11/show-images-in-grid-inside-jupyter-notebook-using-matplotlib-and-numpy/

    Remove whitespace around figure and preserve given figsize:
    https://github.com/matplotlib/matplotlib/issues/11681

    TODO: 1 shelter image occupies a rectangular region instead of a grid cell
    https://matplotlib.org/stable/gallery/images_contours_and_fields/layer_images.html
    """
    imgs = []
    with open(map_fpath, "r") as rfile:
        nrows = 0
        for line in rfile:
            line = line.strip()
            if not line:
                continue
            ncols = 0
            for entity_id in line:
                img = None  # background
                if entity_id in ID2NAME:
                    entity_name = ID2NAME[entity_id]
                    img_fpath = os.path.join(landmark_dpath, "%s.png" % entity_name)
                    if not os.path.exists(img_fpath):
                        img_fpath = os.path.join(landmark_dpath, "%s.jpeg" % entity_name)
                    img = load_image(img_fpath, LANDMARK_SZ)
                imgs.append(img)
                ncols += 1
            nrows += 1

    fig = plt.figure(figsize=(nrows*LANDMARK_SZ[0], ncols*LANDMARK_SZ[1]), constrained_layout=True)
    img_grid = ImageGrid(fig=fig, rect=111, nrows_ncols=(nrows, ncols), axes_pad=0.1)
    for idx, (ax, img) in enumerate(zip(img_grid, imgs)):
        print("completed: %d / %d grid cells" % (idx+1, nrows*ncols))
        if img is not None:
            ax.imshow(img)
    plt.tight_layout(pad=0)
    plt.savefig(save_fpath)
    # plt.savefig(save_fpath, bbox_inches='tight', pad_inches=0)
    print("completed rendering base map")

    # for entity_name in id2name.values():
    #     img_fpath = os.path.join(landmark_dpath, "%s.png" % entity_name)
    #     img = load_image(img_fpath)
    #     plt.imshow(img)
    #     plt.show()


def load_image(img_fpath, img_size=None):
    """
    Load and reshape an image to proper size for display
    """
    img = Image.open(img_fpath).convert('RGB')
    if img_size:
        img = img.resize((img_size[0], img_size[1]))
    img = np.array(img)
    return img


def add_heat_map(decorated_map_fpath, vis_dpath, edge2loc2prob, ltl_id):
    """
    Overlay classifier heat map over map decorated with landmark images.

    Add grid lines to image:
    https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib

    TODO: add colormap and sidebar
    """
    img = load_image(decorated_map_fpath)

    # Set up figure
    my_dpi = 300.
    fig = plt.figure(figsize=(float(img.shape[0]) / my_dpi, float(img.shape[1]) / my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(111)

    # Remove whitespace around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Set the grid interval: here we use the major tick interval
    map_array = load_map(map_fpath)
    grid_dim = map_array.shape[0]  # assume nrows == ncols
    x_interval, y_interval = img.shape[0] / grid_dim, img.shape[1] / grid_dim
    x_loc = plticker.MultipleLocator(base=x_interval)
    y_loc = plticker.MultipleLocator(base=y_interval)
    ax.xaxis.set_major_locator(x_loc)
    ax.yaxis.set_major_locator(y_loc)

    # Add grid lines
    ax.grid(which='major', axis='both', linestyle='-', color="k", linewidth=1)

    # Add the image
    ax.imshow(img)
    img_grid_fpath = os.path.join(vis_dpath, "map_grid.png")
    plt.savefig(img_grid_fpath)

    # Overlay heat map on map figure
    for edge, loc2prob in edge2loc2prob.items():
        img = load_image(img_grid_fpath)  # load a new grid map for this classifier
        fig = plt.figure(figsize=(float(img.shape[0])/my_dpi, float(img.shape[1])/my_dpi), dpi=my_dpi)
        ax = fig.add_subplot(111)
        ax.imshow(img)
        # ax.set_xticks(np.arange(grid_dim), labels=list(range(grid_dim)))  # y indices
        # ax.set_yticks(np.arange(grid_dim), labels=list(range(grid_dim)))  # x indices
        for loc, prob in loc2prob.items():
            y_loc, x_loc = loc  # Mincraft x, y coordinates map to matplotlib y, x coordinates
            x_low, x_up = x_loc * x_interval, (x_loc + 1) * x_interval
            y_low, y_up = y_loc * y_interval, (y_loc + 1) * y_interval
            x = np.arange(x_low, x_up + 0.01, 0.01)  # 0.01: precision of x/y-interval
            y0 = y_low * np.ones(len(x))
            y1 = y_up * np.ones(len(x))
            ax.fill_between(x, y0, y1, color="r", alpha=0.2 * loc2prob[loc])
        vis_classifier_fpath = os.path.join(vis_dpath, "ltl_%d_edge_%s_after.png" % (ltl_id, edge))
        plt.savefig(vis_classifier_fpath)


def load_map(map_fpath):
    map_array = []
    nrows = 0
    with open(map_fpath, "r") as rfile:
        for line in rfile:
            line = line.strip()
            if not line:
                continue
            ncols = 0
            for entity_id in line:
                if entity_id in ID2NAME:
                    entity_id = entity_id.capitalize()
                else:
                    entity_id = " "
                map_array.append(entity_id)
                ncols += 1
            nrows += 1
    map_array = np.array(map_array)
    map_array.resize((nrows, ncols))
    return map_array


if __name__ == '__main__':
    algos = ["lpopl", "zero_shot_transfer"]
    id2tasks = {
        0: "sequence",
        1: "interleaving",
        2: "safety",
        3: "transfer_sequence",
        4: "transfer_interleaving"
    }  # for reference

    parser = argparse.ArgumentParser(prog="visualize initiation set classifier",
                                     description='Visualize initiation set classifiers of relabeled options.')
    parser.add_argument('--algo', default='lpopl', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algos))
    parser.add_argument('--tasks_id', default=4, type=int,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(id2tasks.keys()))
    parser.add_argument('--map_id', default=0, type=int,
                        help='This parameter identify the map on which relabeling happens')
    parser.add_argument('--ltl_id', default=-1, type=int,
                        help='This parameter identify the relabeled option to visualize. -1 visualize all completed LTLs')
    parser.add_argument('--simple_vis', action="store_true",
                        help='This parameter indicated whether to visualize simple map, not decorated landmark images. Include it in command line to use simple visualization')
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.tasks_id not in id2tasks: raise NotImplementedError(
        "Tasks " + str(id2tasks[args.tasks_id]) + " hasn't been defined yet")
    if not (-1 <= args.map_id < 10): raise NotImplementedError("The map must be a number between -1 and 9")

    classifier_dpath = os.path.join("../tmp_oscar", "task_%d" % args.tasks_id, "map_%d" % args.map_id, "classifier")
    map_fpath = os.path.join("../experiments/maps", "map_%d.txt" % args.map_id)
    vis_dpath = os.path.join("..", "vis", "task_%d" % args.tasks_id, "map_%d" % args.map_id)
    os.makedirs(vis_dpath, exist_ok=True)
    run_visualizer(args.algo, args.ltl_id, classifier_dpath, map_fpath, vis_dpath, args.simple_vis)
