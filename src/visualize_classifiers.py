import argparse
import os
import dill
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from zero_shot_transfer import construct_initiation_set_classifiers

LANDMARK_SZ = (15, 15)
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


def visualize_discrete_classifier(algo, classifier_dpath, ltl_id, map_fpath, landmark_dpath):
    rollout_results_fpath = os.path.join(classifier_dpath, "rollout_results_parallel.pkl")
    if os.path.exists(rollout_results_fpath):
        with open(rollout_results_fpath, "rb") as rfile:
            policy2loc2edge2hits = dill.load(rfile)
        policy2edge2loc2prob = construct_initiation_set_classifiers(classifier_dpath)
    else:
        policy2edge2loc2prob = None
        fnames = os.listdir(classifier_dpath)
        ltl2nstates = defaultdict(int)
        for fname in fnames:
            if "ltl" in fname:
                fpath = os.path.join(classifier_dpath, fname)

    vis_dpath = os.path.join(os.path.dirname(classifier_dpath), "vis")
    os.makedirs(vis_dpath, exist_ok=True)

    # map_array = load_map(map_fpath)
    # print(map_array.shape, map_array)

    minecraft_fpath = os.path.join(vis_dpath, "map.png")
    if not os.path.exists(minecraft_fpath):
        plot_map(map_fpath, landmark_dpath, minecraft_fpath)

    ltl = policy2loc2edge2hits["ltls"][ltl_id]
    edge2loc2prob = policy2edge2loc2prob[ltl]
    for edge, loc2prob in edge2loc2prob.items():
        # overlay heat map on map figure
        vis_fpath = os.path.join(vis_dpath, "ltl_%d_edge_%s_init_set.png" % (ltl_id, edge))


def load_map(map_fpath):
    map_array = []
    nrows = 0
    with open(map_fpath, "r") as rfile:
        for line in rfile:
            # print("row: ", nrows)
            line = line.strip()
            if not line:
                continue
            ncols = 0
            for entity_id in line:
                # print("column: ", ncols)
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


def plot_map(map_fpath, landmark_dpath, save_fpath):
    """
    Read in map_x.txt, plot map figure with minecraft landmarks
    https://kanoki.org/2021/05/11/show-images-in-grid-inside-jupyter-notebook-using-matplotlib-and-numpy/
    TODO: 1 shelter image occupies a rectangular region instead of a grid cell
    """
    imgs = []
    with open(map_fpath, "r") as rfile:
        nrows = 0
        for line in rfile:
            print("row: ", nrows)
            line = line.strip()
            if not line:
                continue
            ncols = 0
            for entity_id in line:
                print("column: ", ncols)
                img = None  # empty background
                if entity_id in ID2NAME:
                    entity_name = ID2NAME[entity_id]
                    img_fpath = os.path.join(landmark_dpath, "%s.png" % entity_name)
                    if not os.path.exists(img_fpath):
                        img_fpath = os.path.join(landmark_dpath, "%s.jpeg" % entity_name)
                    img = load_image(img_fpath, LANDMARK_SZ)
                imgs.append(img)
                ncols += 1
            nrows += 1

    fig = plt.figure(figsize=(nrows*LANDMARK_SZ[0], ncols*LANDMARK_SZ[1]))
    img_grid = ImageGrid(fig=fig, rect=111, nrows_ncols=(nrows, ncols), axes_pad=0.02)
    for idx, (ax, img) in enumerate(zip(img_grid, imgs)):
        print("completed: %d / %d" % (idx+1, nrows*ncols))
        if img is not None:
            ax.imshow(img)
    plt.savefig(save_fpath)

    # for entity_name in id2name.values():
    #     img_fpath = os.path.join(landmark_dpath, "%s.png" % entity_name)
    #     img = load_image(img_fpath)
    #     plt.imshow(img)
    #     plt.show()


def load_image(img_fpath, img_size):
    """
    Load an image and reshape to proper size for display
    """
    img = Image.open(img_fpath).convert('RGB')
    img = img.resize((img_size[0], img_size[1]))
    img = np.array(img)
    return img


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
                        help='This parameter identify the relabeled option to visualize')
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.tasks_id not in id2tasks: raise NotImplementedError(
        "Tasks " + str(id2tasks[args.tasks_id]) + " hasn't been defined yet")
    if not (-1 <= args.map_id < 10): raise NotImplementedError("The map must be a number between -1 and 9")

    classifier_dpath = os.path.join("../tmp/", "task_%d/map_%d" % (args.tasks_id, args.map_id), "classifier")
    map_fpath = os.path.join("../experiments/maps", "map_%d.txt" % args.map_id)
    landmark_dpath = os.path.join("../vis/minecraft")
    visualize_discrete_classifier(args.algo, classifier_dpath, args.ltl_id, map_fpath, landmark_dpath)
