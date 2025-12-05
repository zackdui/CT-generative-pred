# Full Process in Data Creation and Experimentation

## Data Creation 


First I ran save_mapping_from_exams_to_nifti from src.CTFM.data.utils to create metadata/zack_exam_to_nifti.pkl Now from initial_exam

Then I ran full_parquet_creation in src.CTFM.data.create_parquets

Then optionally you could run write_new_nifti_files in processing

I ran write_new_nifti_files in processing and saved 100 transforms to test on

