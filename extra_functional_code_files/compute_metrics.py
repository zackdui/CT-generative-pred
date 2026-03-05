import pandas as pd
import numpy as np

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)

    input_vol = df["Input Volume"]
    output_vol = df["Output Volume"]
    gen_vol = df["Generated Volume"]
    dice = df["Dice Score"]

    # Volume error (generated vs output)
    volume_error = gen_vol - output_vol
    abs_volume_error = volume_error.abs()

    mean_volume_error = volume_error.mean()
    median_volume_error = volume_error.median()

    # MAE
    mae = abs_volume_error.mean()
    median_absolute_error = abs_volume_error.median()

    # RMSE
    rmse = np.sqrt((volume_error ** 2).mean())

    # Growth quantities
    true_growth = output_vol - input_vol
    pred_growth = gen_vol - input_vol

    # Mean Absolute Growth Error
    mean_abs_growth_error = (pred_growth - true_growth).abs().mean()

    # Growth Direction Accuracy
    growth_direction_accuracy = (
        np.sign(true_growth) == np.sign(pred_growth)
    ).mean()

    # R^2
    ss_res = ((output_vol - gen_vol) ** 2).sum()
    ss_tot = ((output_vol - output_vol.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    # Mean percentage error: |gen - out| / input
    percent_error = abs_volume_error / input_vol
    mean_percent_error = percent_error.mean()
    median_percent_error = percent_error.median()

    # Mean percentage error: |gen - out| / output
    percent_error_output = abs_volume_error / output_vol
    mean_percent_error_output = percent_error_output.mean()
    median_percent_error_output = percent_error_output.median()

    # Average dice
    mean_dice = dice.mean()
    median_dice = dice.median()

    print("Mean Volume Error:", mean_volume_error)
    print("Median Volume Error:", median_volume_error)
    print("MAE:", mae)
    print("Median Absolute Volume Error", median_absolute_error)
    print("RMSE:", rmse)
    print()
    print("Mean Absolute Growth Error:", mean_abs_growth_error)
    print("Growth Direction Accuracy:", growth_direction_accuracy)
    print("R^2:", r2)
    print()
    print("Mean Percent Error:", mean_percent_error)
    print("Median Percent Error:", median_percent_error)
    print()
    print("Mean Percent Error (relative to output):", mean_percent_error_output)
    print("Median Percent Error (relative to output):", median_percent_error_output)
    print()
    print("Average Dice Score:", mean_dice)
    print("Median Dice Score:", median_dice)

if __name__ == "__main__":
    # Example usage:
    # compute_metrics("/data/rbg/users/duitz/CT-generative-pred/output_metrics/latent_space_eval.csv")
    # compute_metrics("/data/rbg/users/duitz/CT-generative-pred/output_metrics/raw_image_no_bbox.csv")
    # compute_metrics("/data/rbg/users/duitz/CT-generative-pred/output_metrics/metrics_100_small_bbox_updated.csv")
    compute_metrics("/data/rbg/users/duitz/CT-generative-pred/output_metrics/metrics_90_percent_full_updated.csv")