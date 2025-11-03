import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import seaborn as sns


plt.figure()
lw = 2

def plot_data(filepath, color, legend, show_score=False, line_type='solid'):
	flag = False
	score = []
	label = []
	f = open(filepath, 'r')
	show_score=True
	for record in f:
		if "score_deblurganv2" in f.name or "lmi" in f.name:
			flag = True
			score.append(record.split(' ')[0].replace('\n', ''))
			label.append(record.split(' ')[1].replace('\n', ''))
		else:
			score.append(record.split(' ')[2].replace('\n', ''))
			label.append(record.split(' ')[3].replace('\n', ''))
	f.close()
	y_test = np.asarray(label, dtype=float)
	y_score = np.asarray(score, dtype=float)
	if flag:
		print("here")
		y_test = 1-y_test
	fpr, tpr, thresholds = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)
	eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	# thresh = interp1d(fpr, thresholds)(eer)
	if show_score:
		plt.plot(fpr, tpr, color=color, lw=lw, linestyle=line_type, label=legend+'\n(AUC=%0.4f, EER=%0.4f)' % (roc_auc*100, eer*100))
	else:
		plt.plot(fpr, tpr, color=color, lw=lw, linestyle=line_type, label=legend)

def plot_data2(filepath, color, legend, line):
	label = []
	f = open(filepath, 'r')
	for record in f:
		label.append(record.split(' ')[1].replace('\n', ''))
	f.close()

	y = np.asarray(label, dtype=float)
	print(f"{legend}: {y.mean()}")
	sns.kdeplot(y, label= legend, linestyle = line, color = color)

	# plt.plot(y, color=color, lw=lw, label=legend)

#### #### #### PLOT ROC #### #### ####
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_mix_baseline_ROC.txt',                       'green',  'Normal Decoder',   show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_new_gaussian_sigma3-6_decoder_ROC.txt',      'blue',   'Gaussian Decoder', show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_own blurry_motion_gaussian_average_ROC.txt', 'yellow', 'Mixed Blurry',     show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_Gaussian_Blur_Test_ROC.txt',                          'red',    'Gaussian Blurry',  show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_Motion_Blur_Test_ROC.txt',                            'blue',   'Motion Blurry',    show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_Average_Blur_Test_ROC.txt',                           'yellow', 'Average Blurry',   show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_sharpOnSharp_ROC.txt',                                'blue',   'Sharp on Sharp',   show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_New_Blur_Test_ROC.txt',                                     'red',    'New Blurry',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_final_sharp_decoder_ROC.txt',                               'red',    'Sharp',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_Style_Binary_From_Scratch_ROC.txt',                         'orange', 'BAD, S-Start',     show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_8_binary_ROC.txt',                                          'red',    '8 Binary',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_iitb_ROC.txt',                                              'green',  'iitb',             show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_new_binary_1_ROC.txt',                                      'red',    'new Binary',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_binary_style_ROC.txt',                                      'purple', 'Best Model',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_ORIGINAL_BINARY_ROC.txt',                                   'red',    'BAD',              show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_new_inverse_axisnone_ROC.txt',               'black',  'Binary IDWT',      show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_inverse_non_binary_axisNone_ROC.txt',        'blue',   'Sharp IDWT',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_blurred_new_ROC.txt',                        'red',    'Blurry',           show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_bin_1_ROC.txt',                                             'red',    'bin 1',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_bin_2_ROC.txt',                                             'blue',   'bin 2',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_bin_3_ROC.txt',                                             'purple', 'bin 3',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_bin_4_ROC.txt',                                             'green',  'bin 4',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_just_to_do_it_ROC.txt',                                     'orange', 'BAD VERI',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_BIN_TRY_ROC.txt',                                           'green',  'good binary',      show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_test_set_inverse_axisnone_new_ROC.txt',                      'red',    'idk',              show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_binary_style_ROC.txt',                                       'red',    'BAD IDK',          show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_NEW_BIN_SAME_ROC.txt',                                      'green',  'Style From Start', show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_NEW_BIN_SAME_CS_ROC.txt',                                   'red',    'Our Model',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_NEW_BIN_SAME_CS_ROC.txt',                                    'green',  'FINAL MODEL IDK',  show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/in_scores.txt',                                                  'purple', 'poly',             show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/biocop_files/in_scores_biocop.txt',                              'red',    'Baseline',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_no_resize_ROC.txt',                                   'blue',   '512',              show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_resize_ROC.txt',                                      'green',  '256',              show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/in_scores_ridgebase.txt',                                        'red',    'ridgebase idk baseline',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/vf_scores.txt',                                                  'green',  'ridgebase vf baseline',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_noresize_all_deblur_ROC.txt',                      'black',  'ridgebase deblur',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_AdaAttN_same_ROC.txt',                                      'orange', 'AdaAttN',          show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IITB_full/in_scores_iitb_cgt_wname.txt',                         'red',    'IDK IITB',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IITB_full/vf_scores_iitb_cgt_test.txt',                          'orange', 'VERI original',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/iitb_cropped_ROC.txt',                                           'red',    'VERI crop',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_nocrop_ROC.txt',                                        'green',  'IDK nocrop',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/iitb_files/in_scores_iitb.txt',                                  'black',  'idk less vs less',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/iitb_files/vf_scores.txt',                                       'red',    'veri less vs less',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_nocrop_deblur_ROC.txt',                                 'blue',   'idk deblur vs touch iitb',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_original_ROC.txt',                                    'black',  'IDK BioCop OG ',         show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_deblur_ROC.txt',                                      'orange', 'IDK BioCop Deblur ',         show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/ridgebase_files/vf_scores.txt',                                  'blue',      'VERI Blurry',           show_score=True, line_type="dashed")

# # BioCop synth
# plot_data('/home/admin/AdaAttN/datasets/veri_David_sharpOnSharp_ROC.txt',                                'black',   'VERI Ground Truth',   show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_sharpOnSharp_ROC.txt',                                'black',   'IDK Ground Truth',   show_score=True, line_type="dashed")
# # us
# plot_data('/home/admin/AdaAttN/datasets/veri_NEW_BIN_SAME_CS_ROC.txt',                          'purple',       'VERI Ours',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_NEW_BIN_SAME_CS_ROC.txt',                           'purple',       'IDK Ours',             show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/IDK_veri_sharp_vs_AdaAttN_Gamma_0.9_every_time.txt',                           'green',       'IDK Gamma test',             show_score=True, line_type="dashed")

# plot_data('/home/admin/AdaAttN/datasets/biocop_512_vs._biocop_512.txt',                     'red',    'Biocop OG',       show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/biocop_512_vs.biocop_AdaAttN_Gamma_64_randoms.txt', 'green',  'Biocop rands',    show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/biocop_512_vs_AdaAttN_Gamma_64_set_to_0.9.txt',     'blue',   'Biocop 0.9',      show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/biocop_512_vs_Paper_results_biocop_512.txt',        'orange', 'Biocop Paper OG', show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/biocop_512_vs_AdaAttN_Gamma_64_lin_separate_fixed_weights_again.txt','black', 'Biocop fixed weights', show_score=True, line_type="dashed")


# # amol
# plot_data('/home/admin/AdaAttN/datasets/vf_score_clmixpv15.1_regr_final_test_set_test.txt',   'green',      'VERI Amol',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/in_score_clmixpv15.1_regr_biocop_test2_test.txt',     'green',      'IDK Amol',             show_score=True, line_type="dashed")
# # adaattn
# plot_data('/home/admin/AdaAttN/datasets/veri_AdaAttN_same_ROC.txt',                             'orange',     'VERI AdaAttN',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_adaattn_ROC.txt',                            'orange',     'IDK AdaAttN',          show_score=True, line_type="dashed")
# # adain
# # deblurgan
# # plot_data('/home/admin/AdaAttN/datasets/vf_score_deblurganv2_test.txt',                         'red',        'VERI DeblurGan',       show_score=True)
# # plot_data('/home/admin/AdaAttN/datasets/in_score_deblurganv2_test.txt',                         'red',        'IDK DeblurGan ',       show_score=True, line_type="dashed")
# # blurryplot_data('/home/admin/AdaAttN/datasets/veri_veri_sharp_templates_vs_biocop_deblurgan_2_ROC.txt', 'red',        'VERI DeblurGan NEW ',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_veri_sharp_templates_vs_biocop_deblurgan_2_ROC.txt', 'green',        'VERI DeblurGan ',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_veri_sharp_vs_biocop_deblurgan_2.txt',                'green',        'IDK DeblurGan ',       show_score=True, line_type="dashed")

# plot_data('/home/admin/AdaAttN/datasets/veri_adain_ROC.txt',                                    'red',     'VERI AdaIN',           show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_adain_ROC.txt',                              'red',     'IDK AdaIN',            show_score=True, line_type="dashed")

# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_blurred_new_ROC.txt',               'burlywood',      'VERI Blurry',          show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_synthetic_blurry_biocop_ROC.txt',                   'burlywood',      'IDK Blurry',           show_score=True, line_type="dashed")

# plot_data('/home/admin/AdaAttN/datasets/IDK_veri_sharp_vs_AdaAttN_25%_clean_no_resize_150.txt',                           'orange',       'IDK 25% no resize',             show_score=True, line_type="dashed")

# plot_data('/home/admin/AdaAttN/datasets/IDK_veri_sharp_vs_0.7-3_resize_biocop_synth_deblur.txt',                           'blue',       'IDK 25% yes resize',             show_score=True, line_type="dashed")




# Ridgebase Synth
# # us
# # plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400x640_templates_vs_ridgebase_deblur_2_ROC.txt',        'purple',    'VERI Ours',             show_score=True)
# # plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_deblur_2_ROC.txt',                   'purple',    'IDK Ours',              show_score=True, line_type="dashed")
# # amol
# # plot_data('/home/admin/AdaAttN/datasets/in_scores_ridgebase.txt',                                               'yellow',   'VERI OG',             show_score=True)




# # plot_data('/home/admin/AdaAttN/datasets/vf_scores.txt',                                                         'yellow',   'IDK OG',              show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400x640_templates_vs_ridgebase_400x640_ROC.txt',                   'black',  'VERI Ground Truth',              show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_iitb_ridgebase_400x640.txt',                   'black',  'IDK Ground Truth',              show_score=True, line_type="dashed")


# plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400x640_templates_vs_ridgebase_deblur_500_ROC.txt',   'purple',   'VERI Ours',      show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_deblur.txt',                      'purple',   'IDK Ours',      show_score=True, line_type="dashed")

# # plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_amol_ROC.txt',                       'green',   'IDK Amol',            show_score=True, line_type="dashed")
# # adaattn
# plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400_640_templates_vs_ridgebase_adaattn_1_ROC.txt',       'orange',   'VERI AdaAttN',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_400x640_adaattn_1_ROC.txt',          'orange',   'IDK AdaAttN',         show_score=True, line_type="dashed")
# # adain
# # deblurgan
# # plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400x640_templates_vs_ridgebase_deblurgan_500_ROC.txt',   'cyan',   'VERI DeblurGan',      show_score=True)
# # plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_deblurgan_ROC.txt',                  'cyan',   'IDK DeblurGan',       show_score=True, line_type="dashed")
# # # blurry

# plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400x640_templates_vs_ridgebase_deblurgan_2_ROC.txt',   'green',   'VERI DeblurGan',      show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_deblurgan_2.txt',                  'green',   'IDK DeblurGan',      show_score=True, line_type="dashed")

# plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400x640_templates_vs_ridgebase_400x640_adain_4_ROC.txt', 'red',  'VERI AdaIN',          show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_adain_3_ROC.txt',                    'red',  'IDK AdaIN',           show_score=True, line_type="dashed")


# plot_data('/home/admin/AdaAttN/datasets/veri_ridgebase_400x640_templates_vs_ridgebase_400x640_blur_ROC.txt',    'burlywood',  'VERI Blurry',         show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_ridgebase_400x640__blur_ROC.txt',              'burlywood',  'IDK Blurry',          show_score=True, line_type="dashed")


# # # # # # Original

# plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_iitb_ridgebase_400x640.txt',                   'black',  'IDK Ground Truth',              show_score=True, line_type="dashed")
# # # # # # plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_iitb_ridgebase_enhanced.txt',                  'yellow',  'IDK ENHANCED',              show_score=True, line_type="dashed")

# # plot_data('/home/admin/AdaAttN/datasets/IDK_ridgebase_400x640_vs_iitb_ridgebase_adaattn_5.txt',          'yellow',   'IDK AdaAttN NEEEEEEEW',         show_score=True, line_type="dashed")


# IITB Synth
# us

# plot_data('/home/admin/AdaAttN/datasets/iitb_files/vf_scores.txt',                                    'black',     'VERI Ground Truth',            show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/iitb_files/in_scores_iitb.txt',                               'black',     'IDK Ground Truth',             show_score=True, line_type="dashed")

# plot_data('/home/admin/AdaAttN/datasets/veri_iitb_192x304_vs_iitb_192x304_deblur_ROC.txt',            'purple',      'VERI Ours',          show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_192x304_vs_iitb_192x304_deblur_ROC.txt',             'purple',      'IDK Ours',           show_score=True, line_type="dashed")
# # amol

# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_192x304_vs_iitb_amol_ROC.txt',                       'green',     'IDK Amol',           show_score=True, line_type="dashed")
# # adaattn
# plot_data('/home/admin/AdaAttN/datasets/veri_iitb_192x304_vs_iitb_192x304_adaattn_1_ROC.txt',         'orange',    'VERI AdaAttN ',      show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_192x304_vs_iitb_192x304_adaattn_1_ROC.txt',          'orange',    'IDK AdaAttN ',       show_score=True, line_type="dashed")
# # adain
# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_192x304_vs_iitb_deblurgan_2.txt',                  'green',       'IDK DeblurGan',     show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/veri_iitb_192x304_templates_vs_iitb_deblurgan_2_ROC.txt',   'green',       'VERI DeblurGan',     show_score=True)

# plot_data('/home/admin/AdaAttN/datasets/veri_iitb_192x304_vs_iitb_192x304_adain_2_ROC.txt',           'red',    'VERI AdaIN ',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_192x304_vs_iitb_192x304_adain_1_ROC.txt',            'red',    'IDK AdaIN ',         show_score=True, line_type="dashed")
# # deblurgan
# # plot_data('/home/admin/AdaAttN/datasets/veri_iitb_192x304_templates_vs_iitb_deblurgan_500_ROC.txt',   'red',       'VERI DeblurGan ',    show_score=True)
# # plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_192x304_vs_iitb_deblurgan_ROC.txt',                  'red',       'IDK DeblurGan ',     show_score=True, line_type="dashed")
# # blurry
# plot_data('/home/admin/AdaAttN/datasets/veri_iitb_192x304_vs_iitb_192x304_blurry_ROC.txt',            'burlywood',     'VERI Blurry ',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_iitb_192x304_vs_iitb_192x304_blurry_ROC.txt',             'burlywood',     'IDK Blurry',         show_score=True, line_type="dashed")














#BioCop real
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_crooked_og_ROC.txt',                       'red',    'IDK crooked og ',         show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_crooked_no_resize_deb_ROC.txt',            'green',    'IDK crooked deblur ',         show_score=True, line_type="dashed")
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_crooked_no_resize_new_deb_ROC.txt',        'red',  'OG BioCop crooked same pairs',           show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_crooked_deblur_no_resize_new_deb_ROC.txt', 'green',  'Deblur BioCop crooked same pairs',           show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_30_crop_ROC.txt',                            'black',  '30 crop og',           show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_vs_crooked_ROC.txt',                                   'green',    'og crooked 256 same pairs',     show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_vs_deblur_crooked_ROC.txt',                            'green',  'deb crooked 256 same pairs',    show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_30_crop_vs_crooked_30_crop_ROC.txt',                   'orange', 'og crooked 256x30 same pairs',  show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_30_crop_vs_crooked_deblur_30_crop_ROC.txt',            'blue',   'deb crooked 256x30 same pairs', show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_30_crop_vs_crooked_30_crop_DIFF_PAIRS_ROC.txt',        'red',    'og crooked 256x30 diff pairs',  show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_30_crop_vs_crooked_deblur_30_crop_DIFF_PAIRS_ROC.txt', 'green',  'deb crooked 256x30 diff pairs', show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_biocop_blur_250_vs_biocop_blur_250_crooked_SAME_PAIRS_ROC.txt',            'red',   '250 og same pairs', show_score=True)


# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_vs_kept_256_blur_SAME_PAIRS_ROC.txt',            'red',   'kept 0 crop synth', show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_vs_kept_256_deblur_SAME_PAIRS_ROC.txt',            'green',   'kept 0 crop synth', show_score=True)

# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_30_crop_vs_kept_256_30_crop_blur_SAME_PAIRS_ROC.txt',            'orange',   'kept 30 crop synth', show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/IDK_kept_256_30_crop_vs_kept_256_30_crop_deblur_SAME_PAIRS_ROC.txt',            'blue',   'kept 30 crop synth', show_score=True)


# Ablation
plot_data('/home/admin/AdaAttN/datasets/veri_NEW_BIN_SAME_CS_ROC.txt',                            'purple',   'Our Model',        show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_no_identity_ROC.txt',                                'blue',   'No Identity Loss', show_score=True)
plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_new_inverse_axisnone_ROC.txt',        'green', 'Binary IDWT',      show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_no_content_ROC.txt',                                 'cyan',  'No Content Loss',  show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_mix_baseline_ROC.txt',                'orange', 'No Style Loss',    show_score=True)
plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_inverse_non_binary_axisNone_ROC.txt', 'red',    'Sharp IDWT',       show_score=True)
plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_blurred_new_ROC.txt',                 'burlywood',  'Blurry',           show_score=True)

# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_blurred_new_ROC.txt',                 'black',  'Blurry',           show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_inverse_non_binary_axisNone_ROC.txt', 'red',    'Sharp IDWT',       show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_mix_baseline_ROC.txt',                'orange', 'No Style Loss',    show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_no_content_ROC.txt',                                 'green',  'No Content Loss',  show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_David_test_set_new_inverse_axisnone_ROC.txt',        'purple', 'Binary IDWT',      show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_no_identity_ROC.txt',                                'teal',   'No Identity Loss', show_score=True)
# plot_data('/home/admin/AdaAttN/datasets/veri_NEW_BIN_SAME_CS_ROC.txt',                            'blue',   'Our Model',        show_score=True)

#Quality
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_biocop_no_potatoes_test_blurry_scores.txt',      'blue',   'potats blur', line = "solid")
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_biocop_no_potatoes_test_deb_scores.txt',         'red',   'potats deb', line = "solid")
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_iitb_test_original_scores.txt',                  'blue',   'iitb blur', line = "solid")
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_iitb_test_deb_scores.txt',                       'red',   'iitb deb', line = "solid")
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_original_test_blurry_scores.txt',                'blue',   'synth blur', line = "solid")
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_original_test_deb_scores.txt',                   'red',   'synth deb', line = "solid")
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_ridgebase_test_original_scores.txt',             'blue',   'ridge blur', line = "solid")
# plot_data2('/home/admin/AdaAttN/datasets/quality_scores/nfiq_ridgebase_test_deb_scores.txt',                  'red',   'ridge deb', line = "solid")


#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0, 100.0])
# plt.show()

# plt.xlim([0, 1.0])
# plt.ylim([0, 1.0])

plt.ylim([0.7, 1.0])
plt.xscale('log')

plt.xlabel('FAR', fontsize=12)
plt.ylabel('TAR', fontsize=12)
#plt.title('Model Evaluation: On Touch and Touchless Fingerprint Dataset')
plt.legend(loc="lower right", prop={'size':8})
plt.savefig("ROC", dpi = 500)
plt.show()
#plt.savefig('/home/n-lab/Amol/cnn_verifier/contactless/rocs/vf_v12_vs_v15_final_test_set.png')