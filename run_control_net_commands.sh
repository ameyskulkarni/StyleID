#!/bin/bash

#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.65 --output_path output_gamma_0_65
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.65 --controlnet depth --cn_scale 0.25 --output_path output_gamma_0_65_cn_depth_0_25
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.65 --controlnet depth --cn_scale 0.5 --output_path output_gamma_0_65_cn_depth_0_5
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.65 --controlnet depth --cn_scale 0.75 --output_path output_gamma_0_65_cn_depth_0_75
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.65 --controlnet depth --cn_scale 1.0 --output_path output_gamma_0_65_cn_depth_1_0
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.65 --controlnet depth --cn_scale 1.5 --output_path output_gamma_0_65_cn_depth_1_5



#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.50 --output_path output_gamma_0_50

#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.50 --controlnet depth --cn_scale 0.25 --output_path output_gamma_0_50_cn_depth_0_25
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.50 --controlnet depth --cn_scale 0.5 --output_path output_gamma_0_50_cn_depth_0_5
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.50 --controlnet depth --cn_scale 0.75 --output_path output_gamma_0_50_cn_depth_0_75
#
#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.50 --controlnet depth --cn_scale 1.0 --output_path output_gamma_0_50_cn_depth_1_0

#python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.50 --controlnet depth --cn_scale 1.5 --output_path output_gamma_0_50_cn_depth_1_5



python run_styleid.py --cnt data/cnt --sty data/sty --controlnet canny --cn_scale 0.25 --output_path output_cn_canny_0_25

python run_styleid.py --cnt data/cnt --sty data/sty --controlnet canny --cn_scale 0.5 --output_path output_cn_canny_0_5

python run_styleid.py --cnt data/cnt --sty data/sty --controlnet canny --cn_scale 0.75 --output_path output_cn_canny_0_75

python run_styleid.py --cnt data/cnt --sty data/sty --controlnet canny --cn_scale 1.0 --output_path output_cn_canny_1_0

python run_styleid.py --cnt data/cnt --sty data/sty --controlnet canny --cn_scale 1.5 --output_path output_cn_canny_1_5
























#python3 eval_artfid.py --sty ../data/sty_eval/ --cnt ../data/cnt_eval/ --tar ../output_cn_depth_0_25
#python eval_histogan.py --sty ../data/sty_eval --tar ../output_cn_depth_0_25
#
#
#python3 eval_artfid.py --sty ../data/sty_eval/ --cnt ../data/cnt_eval/ --tar ../output_cn_depth_0_5
#python eval_histogan.py --sty ../data/sty_eval --tar ../output_cn_depth_0_5
#
#
#python3 eval_artfid.py --sty ../data/sty_eval/ --cnt ../data/cnt_eval/ --tar ../output_cn_depth_0_75
#python eval_histogan.py --sty ../data/sty_eval --tar ../output_cn_depth_0_75
#
#
#python3 eval_artfid.py --sty ../data/sty_eval/ --cnt ../data/cnt_eval/ --tar ../output_cn_depth_1_0
#python eval_histogan.py --sty ../data/sty_eval --tar ../output_cn_depth_1_0
#
#
#python3 eval_artfid.py --sty ../data/sty_eval/ --cnt ../data/cnt_eval/ --tar ../output_cn_depth_1_5
#python eval_histogan.py --sty ../data/sty_eval --tar ../output_cn_depth_1_5



