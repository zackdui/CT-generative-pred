import json
import pandas as pd

file_raw = "/data/rbg/users/duitz/CT-generative-pred/metadata/test_raw_data_nodule_original_boxes.json" #"test_raw_data_nodule_original_boxes.json"
file_encoded = "/data/rbg/users/duitz/CT-generative-pred/metadata/test_encoded_data_nodule_original_boxes.json" #"test_encoded_data_nodule_original_boxes.json"
test_parquet = "/data/rbg/users/duitz/CT-generative-pred/metadata/test/paired_nodules.parquet"

with open(file_raw, "r") as f:
    raw_outputs = json.load(f)

with open(file_encoded, "r") as f:
    encoded_outputs = json.load(f)

test_pd = pd.read_parquet(test_parquet)

total_above_30 = 0
total_above_100 = 0
total_looked_at = 0
total_below_150 = 0

for index, line in test_pd.iterrows():

    nodule_id_a = f"{line['nodule_group']}_{line['exam_a']}_{line['exam_idx_a']}"
    nodule_id_b = f"{line['nodule_group']}_{line['exam_b']}_{line['exam_idx_b']}"
    if nodule_id_a not in raw_outputs or nodule_id_b not in raw_outputs:
        continue

    total_looked_at += 1

    raw_volume_a = raw_outputs[nodule_id_a]['nodule_volume']
    raw_volume_b = raw_outputs[nodule_id_b]['nodule_volume']

    encoded_volume_a = encoded_outputs[nodule_id_a]['nodule_volume']
    encoded_volume_b = encoded_outputs[nodule_id_b]['nodule_volume']

    raw_diff = raw_volume_a - raw_volume_b
    encoded_diff = encoded_volume_a - encoded_volume_b

    diff_raw_v_encoded = abs(raw_diff - encoded_diff)
    if diff_raw_v_encoded > 30:
        print(f"Above 30 Nodule ID A: {nodule_id_a}, Raw Diff: {raw_diff}, Encoded Diff: {encoded_diff}")
        total_above_30 += 1

    if diff_raw_v_encoded > 100:
        print(f"Above 100 Nodule ID A: {nodule_id_a}, Raw Diff: {raw_diff}, Encoded Diff: {encoded_diff}")
        total_above_100 += 1

    if raw_volume_a < 150:
        total_below_150 += 1
    if raw_volume_b < 150:
        total_below_150 += 1

print(f"total above 30 {total_above_30}")
print(f"total above 100 {total_above_100}")
print(f"total looked at {total_looked_at}")
print(f"total_below_150 {total_below_150}, total {2*total_looked_at}, precent {total_below_150 / (2 * total_looked_at)}")

total_above_ten = 0
total = 0

# for nodule_id in raw_outputs.keys():
#     if nodule_id in encoded_outputs:
#         total += 1
#         raw_volume = raw_outputs[nodule_id]['nodule_volume']
#         encoded_volume = encoded_outputs[nodule_id]['nodule_volume']
#         # if abs(raw_volume - encoded_volume) > 100:
#         #     print(f"Nodule ID: {nodule_id}, Raw Volume: {raw_volume}, Encoded Volume: {encoded_volume}")
#         if abs(raw_volume - encoded_volume) / (raw_volume if raw_volume > 0 else 1) > 0.3:
#             total_above_ten += 1
#             print(f"Nodule ID: {nodule_id}, Raw Volume: {raw_volume}, Encoded Volume: {encoded_volume}")
# print(f"Total nodules with >20% volume difference: {total_above_ten} out of {total} ({(total_above_ten/total)*100:.2f}%)")
# print(f"Total nodules processed: {total} vs total in list {len(raw_outputs)}")

# now check the relative difference between the two. Meaning for the raw and generated between themselves what is the differnce
# Therefore what is the difference between image a and b for raw and encoded
    
    