import os, csv
import pandas as pd

# folder = "/data/rbg/users/duitz/CT-generative-pred/combine_metrics"
# files = [f for f in os.listdir(folder) if f.endswith(".csv")]

# with open(os.path.join(folder, "metrics_merged.csv"), "w", newline="") as out:
#     writer = None
#     for f in files:
#         with open(os.path.join(folder, f)) as inp:
#             reader = csv.reader(inp)
#             header = next(reader)
#             if writer is None:
#                 writer = csv.writer(out)
#                 writer.writerow(header)
#             writer.writerows(reader)


# csv_path = "/data/rbg/users/duitz/CT-generative-pred/output_metrics/metrics_100_small_bbox.csv"
csv_path = "/data/rbg/users/duitz/CT-generative-pred/combine_metrics/metrics_90_percent_full.csv"

df = pd.read_csv(csv_path)

df["Generated Volume"] = df["Generated Volume"].str.extract(r"tensor\(([-\d\.eE]+)").astype(float)

base, ext = os.path.splitext(csv_path)
new_path = base + "_updated" + ext

df.to_csv(new_path, index=False)

print(f"Saved updated file to: {new_path}")