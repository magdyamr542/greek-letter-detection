from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from constants import model_input_size

# all images
images = [
    {
        "bln_id": 1,
        "date_captured": None,
        "file_name": "./images/homer2/txt1/P.Corn.Inv.MSS.A.101.XIII.jpg",
        "height": 2162,
        "id": 1,
        "img_url": "./images/homer2/txt1/P.Corn.Inv.MSS.A.101.XIII.jpg",
        "license": 2,
        "width": 2524,
    },
    {
        "bln_id": 5888,
        "date_captured": None,
        "file_name": "./images/homer2/txt102/P_17054_R_001.jpg",
        "height": 2234,
        "id": 5888,
        "img_url": "./images/homer2/txt102/P_17054_R_001.jpg",
        "license": 4,
        "width": 1441,
    },
    {
        "bln_id": 6109,
        "date_captured": None,
        "file_name": "./images/homer2/txt104/P_21215_R_3_001.jpg",
        "height": 2360,
        "id": 6109,
        "img_url": "./images/homer2/txt104/P_21215_R_3_001.jpg",
        "license": 4,
        "width": 2352,
    },
    {
        "bln_id": 6184,
        "date_captured": None,
        "file_name": "./images/homer2/txt107/P_Oslo_3_66.jpg",
        "height": 2351,
        "id": 6184,
        "img_url": "./images/homer2/txt107/P_Oslo_3_66.jpg",
        "license": 2,
        "width": 2361,
    },
    {
        "bln_id": 13,
        "date_captured": None,
        "file_name": "./images/homer2/txt11/Sorbonne_inv_2089_verso.jpg",
        "height": 1186,
        "id": 13,
        "img_url": "./images/homer2/txt11/Sorbonne_inv_2089_verso.jpg",
        "license": 2,
        "width": 1780,
    },
    {
        "bln_id": 6220,
        "date_captured": None,
        "file_name": "./images/homer2/txt112/P_Heid_inv_G_1262.jpg",
        "height": 2156,
        "id": 6220,
        "img_url": "./images/homer2/txt112/P_Heid_inv_G_1262.jpg",
        "license": 2,
        "width": 2400,
    },
    {
        "bln_id": 6847,
        "date_captured": None,
        "file_name": "./images/homer2/txt113/BNU_Pgr_2344fr46.jpg",
        "height": 552,
        "id": 6847,
        "img_url": "./images/homer2/txt113/BNU_Pgr_2344fr46.jpg",
        "license": 9,
        "width": 666,
    },
    {
        "bln_id": 6853,
        "date_captured": None,
        "file_name": "./images/homer2/txt114/P_Hamb_graec_696.jpg",
        "height": 3110,
        "id": 6853,
        "img_url": "./images/homer2/txt114/P_Hamb_graec_696.jpg",
        "license": 6,
        "width": 2993,
    },
    {
        "bln_id": 6942,
        "date_captured": None,
        "file_name": "./images/homer2/txt116/P_09813_R_001.jpg",
        "height": 2655,
        "id": 6942,
        "img_url": "./images/homer2/txt116/P_09813_R_001.jpg",
        "license": 4,
        "width": 1173,
    },
    {
        "bln_id": 6973,
        "date_captured": None,
        "file_name": "./images/homer2/txt117/P_09949_R_001.jpg",
        "height": 2391,
        "id": 6973,
        "img_url": "./images/homer2/txt117/P_09949_R_001.jpg",
        "license": 4,
        "width": 2333,
    },
    {
        "bln_id": 6989,
        "date_captured": None,
        "file_name": "./images/homer2/txt118/P_07808_R_001.jpg",
        "height": 2428,
        "id": 6989,
        "img_url": "./images/homer2/txt118/P_07808_R_001.jpg",
        "license": 4,
        "width": 3025,
    },
    {
        "bln_id": 7008,
        "date_captured": None,
        "file_name": "./images/homer2/txt119/G_02317_26742_Pap.jpg",
        "height": 2319,
        "id": 7008,
        "img_url": "./images/homer2/txt119/G_02317_26742_Pap.jpg",
        "license": 1,
        "width": 2352,
    },
    {
        "bln_id": 7140,
        "date_captured": None,
        "file_name": "./images/homer2/txt120/P_Oxy_56_3826_recto_verso_unclear.jpg",
        "height": 808,
        "id": 7140,
        "img_url": "./images/homer2/txt120/P_Oxy_56_3826_recto_verso_unclear.jpg",
        "license": 9,
        "width": 1771,
    },
    {
        "bln_id": 7141,
        "date_captured": None,
        "file_name": "./images/homer2/txt120/P_Oxy_56_3826_verso_recto_unclear.jpg",
        "height": 817,
        "id": 7141,
        "img_url": "./images/homer2/txt120/P_Oxy_56_3826_verso_recto_unclear.jpg",
        "license": 9,
        "width": 1765,
    },
    {
        "bln_id": 7220,
        "date_captured": None,
        "file_name": "./images/homer2/txt121/PSI_XIV_1377r.jpg",
        "height": 2880,
        "id": 7220,
        "img_url": "./images/homer2/txt121/PSI_XIV_1377r.jpg",
        "license": 10,
        "width": 2362,
    },
    {
        "bln_id": 7250,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00001_frame_1.jpg",
        "height": 3177,
        "id": 7250,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00001_frame_1.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7251,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00002_frame_2.jpg",
        "height": 3190,
        "id": 7251,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00002_frame_2.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7252,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00003_frame_3.jpg",
        "height": 3184,
        "id": 7252,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00003_frame_3.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7260,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00004_frame_4.jpg",
        "height": 3184,
        "id": 7260,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00004_frame_4.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7254,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00005_frame_5.jpg",
        "height": 3184,
        "id": 7254,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00005_frame_5.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7261,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00006_frame_6.jpg",
        "height": 3177,
        "id": 7261,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00006_frame_6.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7262,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00007_frame_7.jpg",
        "height": 3178,
        "id": 7262,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00007_frame_7.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7264,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8.jpg",
        "height": 3183,
        "id": 7264,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7258,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00009_frame_9.jpg",
        "height": 3177,
        "id": 7258,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00009_frame_9.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 7259,
        "date_captured": None,
        "file_name": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00010_frame_10.jpg",
        "height": 3190,
        "id": 7259,
        "img_url": "./images/homer2/txt122/Bodleian_Library_MS_Gr_class_a_1_P_1_10_00010_frame_10.jpg",
        "license": 3,
        "width": 5344,
    },
    {
        "bln_id": 8145,
        "date_captured": None,
        "file_name": "./images/homer2/txt124/p_bas_27.d.r.jpg",
        "height": 3162,
        "id": 8145,
        "img_url": "./images/homer2/txt124/p_bas_27.d.r.jpg",
        "license": 1,
        "width": 3433,
    },
    {
        "bln_id": 8312,
        "date_captured": None,
        "file_name": "./images/homer2/txt125/p_bas_27.b.r.jpg",
        "height": 2202,
        "id": 8312,
        "img_url": "./images/homer2/txt125/p_bas_27.b.r.jpg",
        "license": 1,
        "width": 1657,
    },
    {
        "bln_id": 8313,
        "date_captured": None,
        "file_name": "./images/homer2/txt125/p_bas_27.b.v.jpg",
        "height": 2214,
        "id": 8313,
        "img_url": "./images/homer2/txt125/p_bas_27.b.v.jpg",
        "license": 1,
        "width": 1711,
    },
    {
        "bln_id": 8466,
        "date_captured": None,
        "file_name": "./images/homer2/txt126/MS._Gr._class._d._41_(P)r.jpg",
        "height": 8160,
        "id": 8466,
        "img_url": "./images/homer2/txt126/MS._Gr._class._d._41_(P)r.jpg",
        "license": 7,
        "width": 10880,
    },
    {
        "bln_id": 9902,
        "date_captured": None,
        "file_name": "./images/homer2/txt130/MS._Gr._class._f._42_(P)r.jpg",
        "height": 8748,
        "id": 9902,
        "img_url": "./images/homer2/txt130/MS._Gr._class._f._42_(P)r.jpg",
        "license": 3,
        "width": 8267,
    },
    {
        "bln_id": 10018,
        "date_captured": None,
        "file_name": "./images/homer2/txt131/901045_0019.jpg",
        "height": 7317,
        "id": 10018,
        "img_url": "./images/homer2/txt131/901045_0019.jpg",
        "license": 3,
        "width": 9116,
    },
    {
        "bln_id": 10178,
        "date_captured": None,
        "file_name": "./images/homer2/txt133/901045_0007.jpg",
        "height": 6507,
        "id": 10178,
        "img_url": "./images/homer2/txt133/901045_0007.jpg",
        "license": 3,
        "width": 7717,
    },
    {
        "bln_id": 10196,
        "date_captured": None,
        "file_name": "./images/homer2/txt134/901045_0009.jpg",
        "height": 7646,
        "id": 10196,
        "img_url": "./images/homer2/txt134/901045_0009.jpg",
        "license": 7,
        "width": 7465,
    },
    {
        "bln_id": 10218,
        "date_captured": None,
        "file_name": "./images/homer2/txt136/MS._Gr._class._e._126_(P)r.jpg",
        "height": 10171,
        "id": 10218,
        "img_url": "./images/homer2/txt136/MS._Gr._class._e._126_(P)r.jpg",
        "license": 7,
        "width": 8295,
    },
    {
        "bln_id": 10304,
        "date_captured": None,
        "file_name": "./images/homer2/txt137/MS._Gr._class._g._49_(P)r.jpg",
        "height": 8160,
        "id": 10304,
        "img_url": "./images/homer2/txt137/MS._Gr._class._g._49_(P)r.jpg",
        "license": 7,
        "width": 10880,
    },
    {
        "bln_id": 10305,
        "date_captured": None,
        "file_name": "./images/homer2/txt137/MS._Gr._class._g._49_(P)v.jpg",
        "height": 8160,
        "id": 10305,
        "img_url": "./images/homer2/txt137/MS._Gr._class._g._49_(P)v.jpg",
        "license": 7,
        "width": 10880,
    },
    {
        "bln_id": 10321,
        "date_captured": None,
        "file_name": "./images/homer2/txt138/P_18125_R_001.jpg",
        "height": 2481,
        "id": 10321,
        "img_url": "./images/homer2/txt138/P_18125_R_001.jpg",
        "license": 4,
        "width": 912,
    },
    {
        "bln_id": 10336,
        "date_captured": None,
        "file_name": "./images/homer2/txt139/P_17211_R_2_001.jpg",
        "height": 3434,
        "id": 10336,
        "img_url": "./images/homer2/txt139/P_17211_R_2_001.jpg",
        "license": 4,
        "width": 3467,
    },
    {
        "bln_id": 10361,
        "date_captured": None,
        "file_name": "./images/homer2/txt140/P_21121_R_3_001.jpg",
        "height": 1421,
        "id": 10361,
        "img_url": "./images/homer2/txt140/P_21121_R_3_001.jpg",
        "license": 4,
        "width": 994,
    },
    {
        "bln_id": 10363,
        "date_captured": None,
        "file_name": "./images/homer2/txt141/P_17002_R_001.jpg",
        "height": 4543,
        "id": 10363,
        "img_url": "./images/homer2/txt141/P_17002_R_001.jpg",
        "license": 4,
        "width": 4701,
    },
    {
        "bln_id": 10410,
        "date_captured": None,
        "file_name": "./images/homer2/txt142/P_Gen_Inv_093r.jpg",
        "height": 1312,
        "id": 10410,
        "img_url": "./images/homer2/txt142/P_Gen_Inv_093r.jpg",
        "license": 5,
        "width": 2000,
    },
    {
        "bln_id": 10425,
        "date_captured": None,
        "file_name": "./images/homer2/txt143/P_Mich_inv_1575.jpg",
        "height": 4895,
        "id": 10425,
        "img_url": "./images/homer2/txt143/P_Mich_inv_1575.jpg",
        "license": 9,
        "width": 4647,
    },
    {
        "bln_id": 10457,
        "date_captured": None,
        "file_name": "./images/homer2/txt144/P_Oxy_6_949_equal_Graz_Ms._I_1954.jpg",
        "height": 1980,
        "id": 10457,
        "img_url": "./images/homer2/txt144/P_Oxy_6_949_equal_Graz_Ms._I_1954.jpg",
        "license": 1,
        "width": 1149,
    },
    {
        "bln_id": 10474,
        "date_captured": None,
        "file_name": "./images/homer2/txt145/P_Oxy_49_3439.jpg",
        "height": 2123,
        "id": 10474,
        "img_url": "./images/homer2/txt145/P_Oxy_49_3439.jpg",
        "license": 1,
        "width": 1349,
    },
    {
        "bln_id": 10617,
        "date_captured": None,
        "file_name": "./images/homer2/txt147/P_Koln_I_21_inv_00046_b_verso.jpg",
        "height": 1084,
        "id": 10617,
        "img_url": "./images/homer2/txt147/P_Koln_I_21_inv_00046_b_verso.jpg",
        "license": 3,
        "width": 741,
    },
    {
        "bln_id": 10618,
        "date_captured": None,
        "file_name": "./images/homer2/txt147/P_Koln_I_21_inv_00046_c_d_verso.jpg",
        "height": 1101,
        "id": 10618,
        "img_url": "./images/homer2/txt147/P_Koln_I_21_inv_00046_c_d_verso.jpg",
        "license": 3,
        "width": 1486,
    },
    {
        "bln_id": 10619,
        "date_captured": None,
        "file_name": "./images/homer2/txt147/P_Koln_I_21_inv_00046_e_verso.jpg",
        "height": 1788,
        "id": 10619,
        "img_url": "./images/homer2/txt147/P_Koln_I_21_inv_00046_e_verso.jpg",
        "license": 3,
        "width": 1393,
    },
    {
        "bln_id": 10620,
        "date_captured": None,
        "file_name": "./images/homer2/txt147/P_Koln_I_21_inv_1030_verso.JPG",
        "height": 3443,
        "id": 10620,
        "img_url": "./images/homer2/txt147/P_Koln_I_21_inv_1030_verso.JPG",
        "license": 3,
        "width": 1350,
    },
    {
        "bln_id": 10699,
        "date_captured": None,
        "file_name": "./images/homer2/txt148/P_Hamb_graec_780.jpg",
        "height": 2664,
        "id": 10699,
        "img_url": "./images/homer2/txt148/P_Hamb_graec_780.jpg",
        "license": 6,
        "width": 2861,
    },
    {
        "bln_id": 10713,
        "date_captured": None,
        "file_name": "./images/homer2/txt149/P_Koln_I_30.JPG",
        "height": 2320,
        "id": 10713,
        "img_url": "./images/homer2/txt149/P_Koln_I_30.JPG",
        "license": 3,
        "width": 3074,
    },
    {
        "bln_id": 10728,
        "date_captured": None,
        "file_name": "./images/homer2/txt150/P_Mich_inv_1210_1216a.jpg",
        "height": 4795,
        "id": 10728,
        "img_url": "./images/homer2/txt150/P_Mich_inv_1210_1216a.jpg",
        "license": 2,
        "width": 2642,
    },
    {
        "bln_id": 10741,
        "date_captured": None,
        "file_name": "./images/homer2/txt151/P_Heid_inv_G_807.jpg",
        "height": 1244,
        "id": 10741,
        "img_url": "./images/homer2/txt151/P_Heid_inv_G_807.jpg",
        "license": 8,
        "width": 407,
    },
    {
        "bln_id": 10752,
        "date_captured": None,
        "file_name": "./images/homer2/txt152/P_Oxy_36_2748.jpg",
        "height": 2242,
        "id": 10752,
        "img_url": "./images/homer2/txt152/P_Oxy_36_2748.jpg",
        "license": 2,
        "width": 1330,
    },
    {
        "bln_id": 10817,
        "date_captured": None,
        "file_name": "./images/homer2/txt153/PSI_VIII_978r.jpg",
        "height": 2230,
        "id": 10817,
        "img_url": "./images/homer2/txt153/PSI_VIII_978r.jpg",
        "license": 10,
        "width": 1656,
    },
    {
        "bln_id": 10846,
        "date_captured": None,
        "file_name": "./images/homer2/txt154/P_Oslo_3_67.jpg",
        "height": 1448,
        "id": 10846,
        "img_url": "./images/homer2/txt154/P_Oslo_3_67.jpg",
        "license": 2,
        "width": 2065,
    },
    {
        "bln_id": 10853,
        "date_captured": None,
        "file_name": "./images/homer2/txt155/Sorbonne_inv_830_verso.jpg",
        "height": 2589,
        "id": 10853,
        "img_url": "./images/homer2/txt155/Sorbonne_inv_830_verso.jpg",
        "license": 1,
        "width": 2424,
    },
    {
        "bln_id": 10882,
        "date_captured": None,
        "file_name": "./images/homer2/txt156/P_Koln_VII_301.jpg",
        "height": 1226,
        "id": 10882,
        "img_url": "./images/homer2/txt156/P_Koln_VII_301.jpg",
        "license": 2,
        "width": 2130,
    },
    {
        "bln_id": 10901,
        "date_captured": None,
        "file_name": "./images/homer2/txt158/P_Med_2_14_pl_1.jpg",
        "height": 3563,
        "id": 10901,
        "img_url": "./images/homer2/txt158/P_Med_2_14_pl_1.jpg",
        "license": 1,
        "width": 1368,
    },
    {
        "bln_id": 10917,
        "date_captured": None,
        "file_name": "./images/homer2/txt159/Sorbonne_inv_2010.jpg",
        "height": 1161,
        "id": 10917,
        "img_url": "./images/homer2/txt159/Sorbonne_inv_2010.jpg",
        "license": 2,
        "width": 1354,
    },
    {
        "bln_id": 10929,
        "date_captured": None,
        "file_name": "./images/homer2/txt160/P_Koln_I_20.JPG",
        "height": 1155,
        "id": 10929,
        "img_url": "./images/homer2/txt160/P_Koln_I_20.JPG",
        "license": 3,
        "width": 1194,
    },
    {
        "bln_id": 10938,
        "date_captured": None,
        "file_name": "./images/homer2/txt161/P_Koln_I_27.JPG",
        "height": 2556,
        "id": 10938,
        "img_url": "./images/homer2/txt161/P_Koln_I_27.JPG",
        "license": 3,
        "width": 2600,
    },
    {
        "bln_id": 10988,
        "date_captured": None,
        "file_name": "./images/homer2/txt161/P_Koln_I_27fr.jpg",
        "height": 654,
        "id": 10988,
        "img_url": "./images/homer2/txt161/P_Koln_I_27fr.jpg",
        "license": 3,
        "width": 640,
    },
    {
        "bln_id": 10989,
        "date_captured": None,
        "file_name": "./images/homer2/txt162/P_Koln_I_29.JPG",
        "height": 2435,
        "id": 10989,
        "img_url": "./images/homer2/txt162/P_Koln_I_29.JPG",
        "license": 3,
        "width": 1244,
    },
    {
        "bln_id": 11010,
        "date_captured": None,
        "file_name": "./images/homer2/txt163/PSI_XIV_1378r.jpg",
        "height": 2880,
        "id": 11010,
        "img_url": "./images/homer2/txt163/PSI_XIV_1378r.jpg",
        "license": 10,
        "width": 1654,
    },
    {
        "bln_id": 11029,
        "date_captured": None,
        "file_name": "./images/homer2/txt164/G_26730_26745_29816_bis_recto.jpg",
        "height": 9321,
        "id": 11029,
        "img_url": "./images/homer2/txt164/G_26730_26745_29816_bis_recto.jpg",
        "license": 1,
        "width": 5332,
    },
    {
        "bln_id": 11134,
        "date_captured": None,
        "file_name": "./images/homer2/txt165/G_26732_Pap.jpg",
        "height": 2548,
        "id": 11134,
        "img_url": "./images/homer2/txt165/G_26732_Pap.jpg",
        "license": 4,
        "width": 2412,
    },
    {
        "bln_id": 11149,
        "date_captured": None,
        "file_name": "./images/homer2/txt166/P_11522_V_3_001.jpg",
        "height": 3242,
        "id": 11149,
        "img_url": "./images/homer2/txt166/P_11522_V_3_001.jpg",
        "license": 4,
        "width": 1555,
    },
    {
        "bln_id": 11170,
        "date_captured": None,
        "file_name": "./images/homer2/txt167/P_Mich_inv_12.jpg",
        "height": 4677,
        "id": 11170,
        "img_url": "./images/homer2/txt167/P_Mich_inv_12.jpg",
        "license": 2,
        "width": 3186,
    },
    {
        "bln_id": 2843,
        "date_captured": None,
        "file_name": "./images/homer2/txt17/P_10574_R_001.jpg",
        "height": 2422,
        "id": 2843,
        "img_url": "./images/homer2/txt17/P_10574_R_001.jpg",
        "license": 4,
        "width": 1908,
    },
    {
        "bln_id": 11683,
        "date_captured": None,
        "file_name": "./images/homer2/txt170/P_Mich_inv_6232.jpg",
        "height": 3777,
        "id": 11683,
        "img_url": "./images/homer2/txt170/P_Mich_inv_6232.jpg",
        "license": 9,
        "width": 4636,
    },
    {
        "bln_id": 11684,
        "date_captured": None,
        "file_name": "./images/homer2/txt171/P_Mich_inv_13.jpg",
        "height": 4724,
        "id": 11684,
        "img_url": "./images/homer2/txt171/P_Mich_inv_13.jpg",
        "license": 2,
        "width": 2931,
    },
    {
        "bln_id": 11725,
        "date_captured": None,
        "file_name": "./images/homer2/txt173/POxy.v0052.n3662.jpg",
        "height": 1337,
        "id": 11725,
        "img_url": "./images/homer2/txt173/POxy.v0052.n3662.jpg",
        "license": 9,
        "width": 2093,
    },
    {
        "bln_id": 2858,
        "date_captured": None,
        "file_name": "./images/homer2/txt18/BNU_Pgr1876_r.jpg",
        "height": 1891,
        "id": 2858,
        "img_url": "./images/homer2/txt18/BNU_Pgr1876_r.jpg",
        "license": 1,
        "width": 1942,
    },
    {
        "bln_id": 2,
        "date_captured": None,
        "file_name": "./images/homer2/txt2/P_CtYBR_inv_69.jpg",
        "height": 1100,
        "id": 2,
        "img_url": "./images/homer2/txt2/P_CtYBR_inv_69.jpg",
        "license": 2,
        "width": 1274,
    },
    {
        "bln_id": 2866,
        "date_captured": None,
        "file_name": "./images/homer2/txt20/G_03085_Pap_recto.jpg",
        "height": 2420,
        "id": 2866,
        "img_url": "./images/homer2/txt20/G_03085_Pap_recto.jpg",
        "license": 1,
        "width": 1884,
    },
    {
        "bln_id": 2886,
        "date_captured": None,
        "file_name": "./images/homer2/txt22/P_06845_R_001.jpg",
        "height": 3014,
        "id": 2886,
        "img_url": "./images/homer2/txt22/P_06845_R_001.jpg",
        "license": 4,
        "width": 2803,
    },
    {
        "bln_id": 4959,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_06869_Z_131ff_R_001.jpg",
        "height": 5482,
        "id": 4959,
        "img_url": "./images/homer2/txt25/P_06869_Z_131ff_R_001.jpg",
        "license": 4,
        "width": 2463,
    },
    {
        "bln_id": 4960,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_06869_Z_494ff_R_001.jpg",
        "height": 6071,
        "id": 4960,
        "img_url": "./images/homer2/txt25/P_06869_Z_494ff_R_001.jpg",
        "license": 4,
        "width": 5024,
    },
    {
        "bln_id": 4958,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_06869_Z_54ff_R_001.jpg",
        "height": 1805,
        "id": 4958,
        "img_url": "./images/homer2/txt25/P_06869_Z_54ff_R_001.jpg",
        "license": 4,
        "width": 1597,
    },
    {
        "bln_id": 4961,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_06869_Z_602ff_R_001.jpg",
        "height": 1677,
        "id": 4961,
        "img_url": "./images/homer2/txt25/P_06869_Z_602ff_R_001.jpg",
        "license": 4,
        "width": 1878,
    },
    {
        "bln_id": 4962,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_07492_R_001.jpg",
        "height": 2630,
        "id": 4962,
        "img_url": "./images/homer2/txt25/P_07492_R_001.jpg",
        "license": 4,
        "width": 2940,
    },
    {
        "bln_id": 4963,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_07493_R_001.jpg",
        "height": 3385,
        "id": 4963,
        "img_url": "./images/homer2/txt25/P_07493_R_001.jpg",
        "license": 4,
        "width": 1883,
    },
    {
        "bln_id": 4964,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_07494_R_001.jpg",
        "height": 2434,
        "id": 4964,
        "img_url": "./images/homer2/txt25/P_07494_R_001.jpg",
        "license": 4,
        "width": 2621,
    },
    {
        "bln_id": 4965,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_07495_R_001.jpg",
        "height": 3396,
        "id": 4965,
        "img_url": "./images/homer2/txt25/P_07495_R_001.jpg",
        "license": 4,
        "width": 2127,
    },
    {
        "bln_id": 2971,
        "date_captured": None,
        "file_name": "./images/homer2/txt25/P_Gen_Inv_095r.jpg",
        "height": 2000,
        "id": 2971,
        "img_url": "./images/homer2/txt25/P_Gen_Inv_095r.jpg",
        "license": 5,
        "width": 1312,
    },
    {
        "bln_id": 3000,
        "date_captured": None,
        "file_name": "./images/homer2/txt28/P_Gen_Inv_085r.jpg",
        "height": 2000,
        "id": 3000,
        "img_url": "./images/homer2/txt28/P_Gen_Inv_085r.jpg",
        "license": 5,
        "width": 1312,
    },
    {
        "bln_id": 3063,
        "date_captured": None,
        "file_name": "./images/homer2/txt29/P_Koln_I_38.JPG",
        "height": 1008,
        "id": 3063,
        "img_url": "./images/homer2/txt29/P_Koln_I_38.JPG",
        "license": 3,
        "width": 1638,
    },
    {
        "bln_id": 3082,
        "date_captured": None,
        "file_name": "./images/homer2/txt30/P_Hamb_graec_665.jpg",
        "height": 5183,
        "id": 3082,
        "img_url": "./images/homer2/txt30/P_Hamb_graec_665.jpg",
        "license": 6,
        "width": 2607,
    },
    {
        "bln_id": 3148,
        "date_captured": None,
        "file_name": "./images/homer2/txt31/P_Heid_inv_G_675=P_Heid_4_289.jpg",
        "height": 868,
        "id": 3148,
        "img_url": "./images/homer2/txt31/P_Heid_inv_G_675=P_Heid_4_289.jpg",
        "license": 8,
        "width": 387,
    },
    {
        "bln_id": 3172,
        "date_captured": None,
        "file_name": "./images/homer2/txt33/P_Mich_inv_1318v.jpg",
        "height": 4654,
        "id": 3172,
        "img_url": "./images/homer2/txt33/P_Mich_inv_1318v.jpg",
        "license": 9,
        "width": 3812,
    },
    {
        "bln_id": 3208,
        "date_captured": None,
        "file_name": "./images/homer2/txt35/P_Mich_inv_1218.jpg",
        "height": 4707,
        "id": 3208,
        "img_url": "./images/homer2/txt35/P_Mich_inv_1218.jpg",
        "license": 2,
        "width": 2754,
    },
    {
        "bln_id": 3273,
        "date_captured": None,
        "file_name": "./images/homer2/txt38/G_31936_Pap.jpg",
        "height": 3638,
        "id": 3273,
        "img_url": "./images/homer2/txt38/G_31936_Pap.jpg",
        "license": 1,
        "width": 1814,
    },
    {
        "bln_id": 3286,
        "date_captured": None,
        "file_name": "./images/homer2/txt39/G_39839_Pap.jpg",
        "height": 2654,
        "id": 3286,
        "img_url": "./images/homer2/txt39/G_39839_Pap.jpg",
        "license": 1,
        "width": 2361,
    },
    {
        "bln_id": 3311,
        "date_captured": None,
        "file_name": "./images/homer2/txt41/PSI_XII_1275r.jpg",
        "height": 1662,
        "id": 3311,
        "img_url": "./images/homer2/txt41/PSI_XII_1275r.jpg",
        "license": 10,
        "width": 955,
    },
    {
        "bln_id": 3333,
        "date_captured": None,
        "file_name": "./images/homer2/txt42/P_Oxy_3_557.jpg",
        "height": 1540,
        "id": 3333,
        "img_url": "./images/homer2/txt42/P_Oxy_3_557.jpg",
        "license": 1,
        "width": 1200,
    },
    {
        "bln_id": 3432,
        "date_captured": None,
        "file_name": "./images/homer2/txt45/P_Oxy_3_555.jpg",
        "height": 942,
        "id": 3432,
        "img_url": "./images/homer2/txt45/P_Oxy_3_555.jpg",
        "license": 1,
        "width": 1200,
    },
    {
        "bln_id": 3453,
        "date_captured": None,
        "file_name": "./images/homer2/txt46/P_Oxy_4_764.jpg",
        "height": 2920,
        "id": 3453,
        "img_url": "./images/homer2/txt46/P_Oxy_4_764.jpg",
        "license": 1,
        "width": 1200,
    },
    {
        "bln_id": 3454,
        "date_captured": None,
        "file_name": "./images/homer2/txt47/Brux_Inv_7188.jpg",
        "height": 1601,
        "id": 3454,
        "img_url": "./images/homer2/txt47/Brux_Inv_7188.jpg",
        "license": 1,
        "width": 757,
    },
    {
        "bln_id": 3468,
        "date_captured": None,
        "file_name": "./images/homer2/txt48/Brux_Inv_7161.jpg",
        "height": 2715,
        "id": 3468,
        "img_url": "./images/homer2/txt48/Brux_Inv_7161.jpg",
        "license": 1,
        "width": 1901,
    },
    {
        "bln_id": 3497,
        "date_captured": None,
        "file_name": "./images/homer2/txt49/BNU_Pgr2480_v.jpg",
        "height": 1719,
        "id": 3497,
        "img_url": "./images/homer2/txt49/BNU_Pgr2480_v.jpg",
        "license": 1,
        "width": 894,
    },
    {
        "bln_id": 5,
        "date_captured": None,
        "file_name": "./images/homer2/txt5/PSI_XIII_1298_15a_r_1.jpg",
        "height": 1815,
        "id": 5,
        "img_url": "./images/homer2/txt5/PSI_XIII_1298_15a_r_1.jpg",
        "license": 10,
        "width": 1584,
    },
    {
        "bln_id": 3512,
        "date_captured": None,
        "file_name": "./images/homer2/txt50/Brux_Inv_5937.jpg",
        "height": 3159,
        "id": 3512,
        "img_url": "./images/homer2/txt50/Brux_Inv_5937.jpg",
        "license": 1,
        "width": 929,
    },
    {
        "bln_id": 3579,
        "date_captured": None,
        "file_name": "./images/homer2/txt52/P_11761_R_4_001.jpg",
        "height": 3258,
        "id": 3579,
        "img_url": "./images/homer2/txt52/P_11761_R_4_001.jpg",
        "license": 4,
        "width": 2596,
    },
    {
        "bln_id": 3628,
        "date_captured": None,
        "file_name": "./images/homer2/txt54/P_Oxy_3_558.jpg",
        "height": 1019,
        "id": 3628,
        "img_url": "./images/homer2/txt54/P_Oxy_3_558.jpg",
        "license": 1,
        "width": 1200,
    },
    {
        "bln_id": 3679,
        "date_captured": None,
        "file_name": "./images/homer2/txt56/P_21242_R_001.jpg",
        "height": 2510,
        "id": 3679,
        "img_url": "./images/homer2/txt56/P_21242_R_001.jpg",
        "license": 4,
        "width": 1761,
    },
    {
        "bln_id": 3711,
        "date_captured": None,
        "file_name": "./images/homer2/txt59/PSI_XIV_1375r.jpg",
        "height": 1936,
        "id": 3711,
        "img_url": "./images/homer2/txt59/PSI_XIV_1375r.jpg",
        "license": 10,
        "width": 1856,
    },
    {
        "bln_id": 3712,
        "date_captured": None,
        "file_name": "./images/homer2/txt59/PSI_XIV_1375v.jpg",
        "height": 1967,
        "id": 3712,
        "img_url": "./images/homer2/txt59/PSI_XIV_1375v.jpg",
        "license": 10,
        "width": 2048,
    },
    {
        "bln_id": 3736,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_a.jpg",
        "height": 1611,
        "id": 3736,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_a.jpg",
        "license": 1,
        "width": 1561,
    },
    {
        "bln_id": 3737,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_b.jpg",
        "height": 257,
        "id": 3737,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_b.jpg",
        "license": 1,
        "width": 327,
    },
    {
        "bln_id": 3738,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_c.jpg",
        "height": 2133,
        "id": 3738,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_c.jpg",
        "license": 1,
        "width": 1829,
    },
    {
        "bln_id": 3739,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_d.jpg",
        "height": 1165,
        "id": 3739,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_d.jpg",
        "license": 1,
        "width": 1345,
    },
    {
        "bln_id": 3740,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_e.jpg",
        "height": 1833,
        "id": 3740,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_e.jpg",
        "license": 1,
        "width": 1585,
    },
    {
        "bln_id": 3741,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_f.jpg",
        "height": 1734,
        "id": 3741,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_f.jpg",
        "license": 1,
        "width": 2522,
    },
    {
        "bln_id": 3742,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_g.jpg",
        "height": 1618,
        "id": 3742,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_g.jpg",
        "license": 1,
        "width": 877,
    },
    {
        "bln_id": 3743,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_h.jpg",
        "height": 1462,
        "id": 3743,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_h.jpg",
        "license": 1,
        "width": 1060,
    },
    {
        "bln_id": 3744,
        "date_captured": None,
        "file_name": "./images/homer2/txt62/P_Oxy_52_3663_i.jpg",
        "height": 1282,
        "id": 3744,
        "img_url": "./images/homer2/txt62/P_Oxy_52_3663_i.jpg",
        "license": 1,
        "width": 955,
    },
    {
        "bln_id": 3914,
        "date_captured": None,
        "file_name": "./images/homer2/txt63/P_Koln_VII_300.jpg",
        "height": 2493,
        "id": 3914,
        "img_url": "./images/homer2/txt63/P_Koln_VII_300.jpg",
        "license": 3,
        "width": 1987,
    },
    {
        "bln_id": 4021,
        "date_captured": None,
        "file_name": "./images/homer2/txt65/P_Koln_II_71.jpg",
        "height": 984,
        "id": 4021,
        "img_url": "./images/homer2/txt65/P_Koln_II_71.jpg",
        "license": 3,
        "width": 988,
    },
    {
        "bln_id": 11608,
        "date_captured": None,
        "file_name": "./images/homer2/txt65/P_Koln_I_26_inv_71_b_c_r.JPG",
        "height": 2252,
        "id": 11608,
        "img_url": "./images/homer2/txt65/P_Koln_I_26_inv_71_b_c_r.JPG",
        "license": 2,
        "width": 2661,
    },
    {
        "bln_id": 11607,
        "date_captured": None,
        "file_name": "./images/homer2/txt65/P_Koln_I_26_inv_71_d_e_r.JPG",
        "height": 2908,
        "id": 11607,
        "img_url": "./images/homer2/txt65/P_Koln_I_26_inv_71_d_e_r.JPG",
        "license": 2,
        "width": 2525,
    },
    {
        "bln_id": 11605,
        "date_captured": None,
        "file_name": "./images/homer2/txt65/P_Koln_I_26_inv_71a_r.JPG",
        "height": 2276,
        "id": 11605,
        "img_url": "./images/homer2/txt65/P_Koln_I_26_inv_71a_r.JPG",
        "license": 2,
        "width": 1203,
    },
    {
        "bln_id": 4025,
        "date_captured": None,
        "file_name": "./images/homer2/txt66/P_Koln_V_207.jpg",
        "height": 3086,
        "id": 4025,
        "img_url": "./images/homer2/txt66/P_Koln_V_207.jpg",
        "license": 3,
        "width": 901,
    },
    {
        "bln_id": 4040,
        "date_captured": None,
        "file_name": "./images/homer2/txt67/P_Koln_I_23_inv_1033_recto.JPG",
        "height": 1718,
        "id": 4040,
        "img_url": "./images/homer2/txt67/P_Koln_I_23_inv_1033_recto.JPG",
        "license": 2,
        "width": 1126,
    },
    {
        "bln_id": 4041,
        "date_captured": None,
        "file_name": "./images/homer2/txt67/P_Koln_I_23_inv_42_recto.JPG",
        "height": 1549,
        "id": 4041,
        "img_url": "./images/homer2/txt67/P_Koln_I_23_inv_42_recto.JPG",
        "license": 2,
        "width": 783,
    },
    {
        "bln_id": 4060,
        "date_captured": None,
        "file_name": "./images/homer2/txt68/P_07807_R_001.jpg",
        "height": 3551,
        "id": 4060,
        "img_url": "./images/homer2/txt68/P_07807_R_001.jpg",
        "license": 4,
        "width": 2467,
    },
    {
        "bln_id": 4166,
        "date_captured": None,
        "file_name": "./images/homer2/txt69/G_12516_a_b_Pap.jpg",
        "height": 2256,
        "id": 4166,
        "img_url": "./images/homer2/txt69/G_12516_a_b_Pap.jpg",
        "license": 1,
        "width": 2910,
    },
    {
        "bln_id": 4191,
        "date_captured": None,
        "file_name": "./images/homer2/txt69/G_12516_c_Pap1.jpg",
        "height": 452,
        "id": 4191,
        "img_url": "./images/homer2/txt69/G_12516_c_Pap1.jpg",
        "license": 1,
        "width": 578,
    },
    {
        "bln_id": 4176,
        "date_captured": None,
        "file_name": "./images/homer2/txt69/G_12516_d_26734_b_Pap.jpg",
        "height": 2416,
        "id": 4176,
        "img_url": "./images/homer2/txt69/G_12516_d_26734_b_Pap.jpg",
        "license": 1,
        "width": 3351,
    },
    {
        "bln_id": 4097,
        "date_captured": None,
        "file_name": "./images/homer2/txt69/G_26734_a_35721.jpg",
        "height": 4533,
        "id": 4097,
        "img_url": "./images/homer2/txt69/G_26734_a_35721.jpg",
        "license": 1,
        "width": 2037,
    },
    {
        "bln_id": 7139,
        "date_captured": None,
        "file_name": "./images/homer2/txt69/G_26734_c.jpg",
        "height": 2158,
        "id": 7139,
        "img_url": "./images/homer2/txt69/G_26734_c.jpg",
        "license": 1,
        "width": 997,
    },
    {
        "bln_id": 4145,
        "date_captured": None,
        "file_name": "./images/homer2/txt69/G_26734_d_Pap.jpg",
        "height": 2642,
        "id": 4145,
        "img_url": "./images/homer2/txt69/G_26734_d_Pap.jpg",
        "license": 1,
        "width": 2304,
    },
    {
        "bln_id": 4158,
        "date_captured": None,
        "file_name": "./images/homer2/txt69/G_26734_e_39833_Pap.jpg",
        "height": 2069,
        "id": 4158,
        "img_url": "./images/homer2/txt69/G_26734_e_39833_Pap.jpg",
        "license": 1,
        "width": 2238,
    },
    {
        "bln_id": 7,
        "date_captured": None,
        "file_name": "./images/homer2/txt7/P_Laur_IV_127r.jpg",
        "height": 2749,
        "id": 7,
        "img_url": "./images/homer2/txt7/P_Laur_IV_127r.jpg",
        "license": 10,
        "width": 2082,
    },
    {
        "bln_id": 4195,
        "date_captured": None,
        "file_name": "./images/homer2/txt71/P_21216_R_3_001.jpg",
        "height": 3782,
        "id": 4195,
        "img_url": "./images/homer2/txt71/P_21216_R_3_001.jpg",
        "license": 4,
        "width": 2501,
    },
    {
        "bln_id": 4223,
        "date_captured": None,
        "file_name": "./images/homer2/txt72/P_09584_R_2_001.jpg",
        "height": 2984,
        "id": 4223,
        "img_url": "./images/homer2/txt72/P_09584_R_2_001.jpg",
        "license": 4,
        "width": 2923,
    },
    {
        "bln_id": 4237,
        "date_captured": None,
        "file_name": "./images/homer2/txt73/BNU_Pgr1242_r.jpg",
        "height": 1729,
        "id": 4237,
        "img_url": "./images/homer2/txt73/BNU_Pgr1242_r.jpg",
        "license": 1,
        "width": 799,
    },
    {
        "bln_id": 4249,
        "date_captured": None,
        "file_name": "./images/homer2/txt74/G_31798_Pap_verso.jpg",
        "height": 4092,
        "id": 4249,
        "img_url": "./images/homer2/txt74/G_31798_Pap_verso.jpg",
        "license": 1,
        "width": 2111,
    },
    {
        "bln_id": 4272,
        "date_captured": None,
        "file_name": "./images/homer2/txt75/P_07507_R_001.jpg",
        "height": 2325,
        "id": 4272,
        "img_url": "./images/homer2/txt75/P_07507_R_001.jpg",
        "license": 4,
        "width": 2329,
    },
    {
        "bln_id": 4279,
        "date_captured": None,
        "file_name": "./images/homer2/txt76/G_26751_Pap.jpg",
        "height": 1848,
        "id": 4279,
        "img_url": "./images/homer2/txt76/G_26751_Pap.jpg",
        "license": 1,
        "width": 1380,
    },
    {
        "bln_id": 4967,
        "date_captured": None,
        "file_name": "./images/homer2/txt78/P_21185_R_3_001.jpg",
        "height": 2989,
        "id": 4967,
        "img_url": "./images/homer2/txt78/P_21185_R_3_001.jpg",
        "license": 4,
        "width": 1137,
    },
    {
        "bln_id": 4981,
        "date_captured": None,
        "file_name": "./images/homer2/txt79/P_Flor_2_107v.jpg",
        "height": 2500,
        "id": 4981,
        "img_url": "./images/homer2/txt79/P_Flor_2_107v.jpg",
        "license": 1,
        "width": 1419,
    },
    {
        "bln_id": 4995,
        "date_captured": None,
        "file_name": "./images/homer2/txt80/P_Koln_IV_181.JPG",
        "height": 1982,
        "id": 4995,
        "img_url": "./images/homer2/txt80/P_Koln_IV_181.JPG",
        "license": 3,
        "width": 1598,
    },
    {
        "bln_id": 5004,
        "date_captured": None,
        "file_name": "./images/homer2/txt81/P_Koln_I_34.JPG",
        "height": 2348,
        "id": 5004,
        "img_url": "./images/homer2/txt81/P_Koln_I_34.JPG",
        "license": 3,
        "width": 752,
    },
    {
        "bln_id": 5071,
        "date_captured": None,
        "file_name": "./images/homer2/txt84/P_11645_R_001.jpg",
        "height": 1860,
        "id": 5071,
        "img_url": "./images/homer2/txt84/P_11645_R_001.jpg",
        "license": 4,
        "width": 1023,
    },
    {
        "bln_id": 5115,
        "date_captured": None,
        "file_name": "./images/homer2/txt86/P_31080_R_001.jpg",
        "height": 2394,
        "id": 5115,
        "img_url": "./images/homer2/txt86/P_31080_R_001.jpg",
        "license": 4,
        "width": 1841,
    },
    {
        "bln_id": 5134,
        "date_captured": None,
        "file_name": "./images/homer2/txt87/P_08440_R_001.jpg",
        "height": 7120,
        "id": 5134,
        "img_url": "./images/homer2/txt87/P_08440_R_001.jpg",
        "license": 4,
        "width": 3915,
    },
    {
        "bln_id": 5178,
        "date_captured": None,
        "file_name": "./images/homer2/txt89/P_18177_R_3_001.jpg",
        "height": 3173,
        "id": 5178,
        "img_url": "./images/homer2/txt89/P_18177_R_3_001.jpg",
        "license": 4,
        "width": 1989,
    },
    {
        "bln_id": 9,
        "date_captured": None,
        "file_name": "./images/homer2/txt9/P_Laur_IV_129r.jpg",
        "height": 1698,
        "id": 9,
        "img_url": "./images/homer2/txt9/P_Laur_IV_129r.jpg",
        "license": 2,
        "width": 1695,
    },
    {
        "bln_id": 10,
        "date_captured": None,
        "file_name": "./images/homer2/txt9/P_Laur_IV_129v.jpg",
        "height": 1576,
        "id": 10,
        "img_url": "./images/homer2/txt9/P_Laur_IV_129v.jpg",
        "license": 2,
        "width": 1722,
    },
    {
        "bln_id": 5671,
        "date_captured": None,
        "file_name": "./images/homer2/txt95/Sorbonne_inv_2303.jpg",
        "height": 1770,
        "id": 5671,
        "img_url": "./images/homer2/txt95/Sorbonne_inv_2303.jpg",
        "license": 2,
        "width": 2164,
    },
    {
        "bln_id": 5685,
        "date_captured": None,
        "file_name": "./images/homer2/txt96/P_09774_R_001.jpg",
        "height": 3747,
        "id": 5685,
        "img_url": "./images/homer2/txt96/P_09774_R_001.jpg",
        "license": 3,
        "width": 3897,
    },
    {
        "bln_id": 5711,
        "date_captured": None,
        "file_name": "./images/homer2/txt97/Sorbonne_inv_542.jpg",
        "height": 1135,
        "id": 5711,
        "img_url": "./images/homer2/txt97/Sorbonne_inv_542.jpg",
        "license": 2,
        "width": 1036,
    },
]


def main():
    # args
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--test-size",
        required=True,
        help="size of testing images relative to all images",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="name of directory to create",
    )
    args = parser.parse_args()
    test_size = int(args.test_size)
    dirname = args.name
    Path(dirname).mkdir(parents=True, exist_ok=True)

    if test_size < 1:
        print("test size cannot be smaller than 1")
        exit(1)

    if test_size > 1 / 2 * len(images):
        print("test size cannot exceed half of total images")
        exit(1)

    print(f"num of images {len(images)}")
    print(f"test size {test_size}")

    train_and_validation, test = train_test_split(
        images, random_state=4, test_size=test_size, shuffle=True
    )

    print(f"num of training and validation images {len(train_and_validation)}")
    print(f"num of testing images {len(test)}")

    def create_coco_file(cocoJson, newImages, filePath):
        cocoJson["images"] = newImages
        with open(filePath, "w") as f:
            json.dump(cocoJson, f)

    base_coco_path = "./data/training/coco.json"
    with open(base_coco_path) as f:
        cocoJson = json.load(f)
        trainPath = os.path.join(dirname, "coco_train.json")
        testPath = os.path.join(dirname, "coco_test.json")

        create_coco_file(cocoJson, train_and_validation, trainPath)
        create_coco_file(cocoJson, test, testPath)
        print(f"created {trainPath}")
        print(f"created {testPath}")

        # create train and test images
        train_images_path = os.path.join(dirname, "training")
        test_images_path = os.path.join(dirname, "testing")

        def copy_images(image_type: str, images_array, images_directory):
            print(f"copying {image_type} images ...")
            for img in images_array:
                path = img["file_name"][:-1]
                dir = "/".join(path.split("/")[:-1])
                # make the dir
                Path(os.path.join(images_directory, dir)).mkdir(
                    parents=True, exist_ok=True
                )
                # copy the image
                src = os.path.join(
                    "data",
                    "training",
                    img["file_name"].replace("images/", "allImages/"),
                )
                dst = os.path.join(images_directory, img["file_name"])
                shutil.copyfile(src, dst)

        copy_images("training", train_and_validation, train_images_path)
        copy_images("testing", test, test_images_path)

        print("create crops for train")
        create_crops_for_train_and_val(dirname)
        print("done crops for train")

        print("create crops for testing")
        create_crops_for_testing(dirname)
        print("done crops for testing")


def create_crops_for_train_and_val(data_dir):
    Path(os.path.join(data_dir, "crops")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_dir, "crops", "train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_dir, "crops", "val")).mkdir(parents=True, exist_ok=True)

    try:
        f = open(os.path.join(data_dir, "coco_train.json"))
    except:
        print("No json file was found!")
        exit(1)

    data = json.load(f)
    f.close()
    l = []
    print(f"creating crops for training with {len(data['images'])} images")
    images_dir = os.path.join(data_dir, "training")
    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        fname = os.path.join(images_dir, img_url)
        image_id = image["bln_id"]
        try:
            Image.open(fname).convert("RGB")
            l.append(image_id)
        except Exception as e:
            print(e)
            continue

    train, val = train_test_split(l, random_state=8)

    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        image_id = image["bln_id"]
        fname = os.path.join(images_dir, img_url)
        try:
            im = Image.open(fname).convert("RGB")
        except:
            print(f"File not found {fname}")
            continue

        if image_id in val:
            split = "val"
        else:
            split = "train"
        for annotation in data["annotations"]:
            if annotation["image_id"] == image_id:
                crop_id = annotation["id"]
                crop_filename = str(image_id) + "_" + str(crop_id) + ".jpg"
                x, y, w, h = annotation["bbox"]

                crop_directory = annotation["category_id"]
                crop_directory = os.path.join(
                    data_dir, "crops", split, str(crop_directory)
                )

                if not os.path.exists(crop_directory):
                    os.mkdir(crop_directory)

                path = os.path.join(crop_directory, crop_filename)
                crop1 = im.crop((x, y, x + w, y + h))
                crop1 = crop1.resize(
                    (model_input_size, model_input_size), Image.Resampling.BILINEAR
                )
                crop1.save(path, "JPEG", quality=85)


def create_crops_for_testing(data_dir: str):
    Path(os.path.join(data_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_dir, "crops", "test")).mkdir(parents=True, exist_ok=True)

    # get the coco.json file
    try:
        f = open(os.path.join(data_dir, "coco_test.json"))
    except:
        print("No json file was found!")
        raise

    data = json.load(f)
    f.close()
    l = []

    print(f"Creating testing crops for {len(data['images'])} images...")
    images_dir = os.path.join(data_dir, "testing")
    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        fname = os.path.join(images_dir, img_url)
        image_id = image["bln_id"]
        try:
            Image.open(fname).convert("RGB")
            l.append(image_id)
        except:
            print(f"Could not open image with id {image_id}")
            raise

    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        image_id = image["bln_id"]
        fname = os.path.join(images_dir, img_url)
        try:
            im = Image.open(fname).convert("RGB")
        except:
            print(f"Image {fname} does not exist")
            raise

        for annotation in data["annotations"]:
            if annotation["image_id"] == image_id:
                crop_id = annotation["id"]
                crop_filename = str(image_id) + "_" + str(crop_id) + ".jpg"
                x, y, w, h = annotation["bbox"]

                crop_directory = annotation["category_id"]
                crop_directory = os.path.join(
                    data_dir, "crops", "test", str(crop_directory)
                )
                if not os.path.exists(crop_directory):
                    os.mkdir(crop_directory)
                path = os.path.join(crop_directory, crop_filename)
                crop1 = im.crop((x, y, x + w, y + h))
                crop1 = crop1.resize(
                    (model_input_size, model_input_size), Image.Resampling.BILINEAR
                )
                crop1.save(path, "JPEG", quality=85)


if __name__ == "__main__":
    main()
