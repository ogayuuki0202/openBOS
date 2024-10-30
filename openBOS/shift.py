from skimage.metrics import structural_similarity as ssm
import numpy as np
import libs as ib

def ssim(ref_array,exp_array):

    # compute the strucutural similarity matrix (SSM) on the grayscaled images
    (score, diff) = ssm(ref_array, exp_array, full=True)
    diff_inv=-diff 
    return  diff_inv


def main_process(ref_array,exp_array):
    ar_ref=ref_array
    ar_exp=exp_array

    #2値化
    bin_ref=ib.biner_thresh(ar_ref, 128)
    bin_exp=ib.biner_thresh(ar_exp, 128)
    
    #二値化した基準画像から色境界の座標を検出
    ref_u,ref_d=ib.bin_indexer(bin_ref)
    ref_u=np.nan_to_num(ref_u)
    ref_d=np.nan_to_num(ref_d)
    
    #二値化した実験画像から色境界の座標を検出,uは白縞の上端,dは白縞の下端
    exp_u,exp_d=ib.bin_indexer(bin_exp)
    exp_u=np.nan_to_num(exp_u)
    exp_d=np.nan_to_num(exp_d)

    #移動量が異常に大きいデータをノイズとして除去
    ref_u,exp_u=ib.noize_reducer_2(ref_u,exp_u,10)
    ref_d,exp_d=ib.noize_reducer_2(ref_d,exp_d,10)
    
  
    #縞の上端と下端のデータを統合して縞の中心を計算
    ref=ib.mixing(ref_u,ref_d)
    exp=ib.mixing(exp_u,exp_d)
    
    #移動量計算(上への移動がプラス)
    diff=-(exp-ref)
    
    #移動量を正しい位置に並べなおして、間を補完
    diff_comp=ib.complementer(ref, diff)
    
    #移動量の平均で除算して、背景全体の移動を打消し
    diff_comp=diff_comp-np.nanmean(diff_comp[0:1000,10:100])

    return diff_comp
    