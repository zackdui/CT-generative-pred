# This file will take the nodule pickle file and convert it to parquet for easier handling
from CTFM.data import pickle_to_parquet


if __name__ == "__main__":
    p_path = "/data/rbg/users/pgmikhael/current/SybilX/notebooks/NLSTNodules/pid2tracked_nodules.p"
    out_parquet = "/data/rbg/users/duitz/CT-generative-pred/metadata/nlst_nodule_tracking.parquet"

    pickle_to_parquet(p_path, out_parquet)