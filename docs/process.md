# Full Process in Data Creation and Experimentation

## Data Creation 


First I ran save_mapping_from_exams_to_nifti is save_mapping to create metadata/zack_exam_to_nifti.pkl 

Then I ran run_create_parquets.py in scripts to create the parquets for train, test, and val.

<!-- Then I ran full_parquet_creation in src.CTFM.data.create_parquets -->

Then optionally you could run write_new_nifti_files in save_registrations
But I just ran it once to create some test files. 
I ran write_registrations for train, test, and val and update parquets for all three splits.

I ran write_new_nifti_files in processing and saved 100 transforms to test on

Ran save_transformed_mapping in save_mapping to creat exam_to_nifti_transformed.pkl in metadata

