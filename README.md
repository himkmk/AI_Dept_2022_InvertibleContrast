# AI_Dept_2022_INVERTIBLE_CONTRAST
__________

This repository contains a simple method to apply invertible image localization for various image reconstruction tasks.



Get Ready
```
pip install -r requirements.txt
```

<img src="./figs/recon_2.png" height="276" width="360"> <img src="./figs/cont_2.png" height="276" width="360">


Easy run
```
# To apply contrast on images
python invertible_contrast.py --run_type contrast --input_path your_input_path --save_path your_save_path

# To get reconstructed images
python invertible_contrast.py --run_type reconstruction  --input_path your_input_path --save_path your_save_path
```

