import os
import pickle

import fire
import matplotlib.pyplot as plt
import numpy as np
from ribs.visualize import grid_archive_heatmap


def main(
    logdir,
    threshold=80,
    meas1_min=-np.inf,
    meas1_max=np.inf,
    meas2_min=-np.inf,
    meas2_max=np.inf,
    mode="elite",
):
    with open(f"{logdir}/reload.pkl", "rb") as f:
        data = pickle.load(f)
        if "archive" in data:
            arch = data["archive"]
        else:
            arch = data["scheduler"].archive

    arch_df = arch.as_pandas()
    print(f"Measure0 range: {arch_df['measure_0'].min()}, {arch_df['measure_0'].max()}")
    print(f"Measure1 range: {arch_df['measure_1'].min()}, {arch_df['measure_1'].max()}")

    if mode == "elite":
        # fig, ax = plt.subplots()
        # grid_archive_heatmap(arch, ax=ax, vmin=0, vmax=100, cmap="viridis")
        # os.makedirs(f"heatmap_figs/{logdir}/", exist_ok=True)
        # fig.savefig(f"heatmap_figs/{logdir}/archive_heatmap.png")

        elite_found = False
        for elite in arch:
            if (
                elite.objective >= threshold
                and meas1_max >= elite.measures[0] >= meas1_min
                and meas2_max >= elite.measures[1] >= meas2_min
            ):
                elite_found = True
                break

        if elite_found:
            np.set_printoptions(precision=3, suppress=True)
            print(
                f"Obj: {elite.objective}; "
                f"Unreg Obj: {elite.metadata['unreg_obj']}; "
                f"Meas: {elite.measures}; "
                f"RepairedSol: {elite.metadata['solution'].tolist()}"
            )
        else:
            print("No such elite")
    elif mode == "reg":
        reg_costs = [elite.metadata["reg_cost"] for elite in arch]
        print(
            f"Mean reg cost: {np.mean(reg_costs)}; "
            f"Std reg cost: {np.std(reg_costs)}; "
            f"Max reg cost: {np.max(reg_costs)}; "
            f"Min reg cost: {np.min(reg_costs)}"
        )


if __name__ == "__main__":
    fire.Fire(main)
