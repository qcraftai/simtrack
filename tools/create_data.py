
from pathlib import Path
import fire
from det3d.datasets.utils.create_gt_database import create_groundtruth_database


def nuscenes_data_prep(root_path, version, nsweeps=10):
    
    import det3d.datasets.nuscenes.nuscenes_tracking as nu_ds
    nu_ds.create_nuscenes_tracking_infos(root_path, version=version, nsweeps=nsweeps)
    
    
    create_groundtruth_database(
        "NUSC",
        root_path,
        Path(root_path) / "infos_train_{:02d}sweeps_tracking.pkl".format(nsweeps),
        nsweeps=1,
    )


if __name__ == "__main__":
    fire.Fire()
