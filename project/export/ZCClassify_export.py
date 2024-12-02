from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-cls.pt")  # load an official model
model = YOLO("runs/train/exp/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")

# {
#     0: 'AAA_3PT_0', 
#     1: 'AAB_3PT_1', 
#     2: 'AAC_20Pin_0', 
#     3: 'AAD_20Pin_1', 
#     4: 'AAE_AC_TongXunXian_0', 
#     5: 'AAF_AC_TongXunXian_1', 
#     6: 'AAG_KaKou_0', 
#     7: 'AAH_KaKou_1', 
#     8: 'AAI_KaKou_2', 
#     9: 'AAJ_KaKou_3', 
#     10: 'AAK_LuoSi_0', 
#     11: 'AAL_LuoSi_1', 
#     12: 'AAM_LuoSi_2', 
#     13: 'AAN_LuoSi_3', 
#     14: 'AAO_LuoSi_5', 
#     15: 'AAP_LuoZhu_0', 
#     16: 'AAQ_LuoZhu_1', 
#     17: 'AAR_PLC_3PT_0', 
#     18: 'AAS_PLC_3PT_1', 
#     19: 'AAT_PXCiHuan_0', 
#     20: 'AAU_TuiZhen_0', 
#     21: 'AAV_TuiZhen_1', 
#     22: 'AAW_XianShu_KunZa_0', 
#     23: 'AAX_XianShu_KunZa_1'
# }