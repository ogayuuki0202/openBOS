import numpy as np
import pandas as pd

def biner_thresh(ar_in,thresh):
    ar_bin=ar_in>thresh
    return ar_bin

def bin_indexer(ar_in):
    #微分処理
    ar_in=ar_in.astype(np.int8)
    ar2=np.delete(ar_in,0,0)
    ar3=np.delete(ar_in , ar_in.shape[0]-1 , 0)
    ar4=ar2-ar3
    
    #微分して+の部分を検出
    u_tuple=np.where(ar4>0)
    #使いやすい形に変更
    u_index=np.stack([u_tuple[1],u_tuple[0]]).T
    
    
    df=pd.DataFrame(u_index)
    df.columns=["X","Y"]

    df=df.pivot(index="Y",columns="X",values="Y")

    df["index"]=df.index/10
    df["index"]=df[["index"]].astype(int)

    df=df.groupby("index").mean()
    df=df/10

    df.columns=range(df.shape[1])
    
    u_index=np.array(df)
    
    #検出y座標のみ並べた表に変換する
    #結果格納用array作成
    #何個あるか分からないのでとりあえず1000個分の表を作成(余った部分はnan)
    u_index_2=np.zeros([1000,ar_in.shape[1]])#.astype(np.int64)
    #一行ずつ処理する方法以外見つかりませんでした
    for x in range(ar_in.shape[1]):

        ar_loop=u_index[:,x][np.where(~np.isnan(u_index[:,x]))[0]]
        
    #微分して-の部分を検出
    d_tuple=np.where(ar4<0)
    #使いやすい形に変更
    d_index=np.stack([d_tuple[1],d_tuple[0]]).T
    
    
    df=pd.DataFrame(d_index)
    df.columns=["X","Y"]

    df=df.pivot(index="Y",columns="X",values="Y")

    df["index"]=df.index/10
    df["index"]=df[["index"]].astype(int)

    df=df.groupby("index").mean()
    df=df/10

    df.columns=range(df.shape[1])
    
    d_index=np.array(df)
    
    #検出y座標のみ並べた表に変換する
    #結果格納用array作成
    #何個あるか分からないのでとりあえず1000個分の表を作成(余った部分はnan)
    d_index_2=np.zeros([1000,ar_in.shape[1]])#.astype(np.int64)
    
    #一行ずつ処理する方法以外見つかりませんでした
    for x in range(ar_in.shape[1]):
        ar_loop=d_index[:,x][np.where(~np.isnan(d_index[:,x]))[0]]
        d_index_2[:,x]=np.concatenate([ar_loop,np.full(1000-ar_loop.shape[0],np.nan)])#1000個丁度になるようにnanで埋める
    
    return u_index_2,d_index_2

def noize_reducer(ar_in):
    test=np.delete(ar_in,0,0)-np.delete(ar_in,ar_in.shape[0]-1,0)
    test2=np.insert(test,test.shape[0],0,axis=0)>np.nanmean(test)*0.7
    
    #縞間隔が平均値の70%以下の場合ノイズと考えられるので弾く
    ar_out=np.zeros([1000,ar_in.shape[1]])
    for x in range(ar_in.shape[1]):
        ar_loop=ar_in[test2[:,x],x]
        ar_out[:,x]=np.concatenate([ar_loop,np.full(1000-ar_loop.shape[0],np.nan)])#1000個丁度になるようにnanで埋める
    
    return ar_out


def noize_reducer_2(ar_ref,ar_exp,diff_thresh):
    for x in range(ar_ref.shape[1]):
        ref=ar_ref[:,x]
        exp=ar_exp[:,x]
        
        while np.any(abs(exp-ref)>diff_thresh):
            
            if np.any(exp-ref>diff_thresh):
                y=np.where(exp-ref>diff_thresh)[0].min()
                ref=np.delete(ref,y)
                ref=np.insert(ref,ref.shape[0],np.nan)
            
            if np.any(exp-ref<-diff_thresh):
                y=np.where(exp-ref<-diff_thresh)[0].min()
                exp=np.delete(exp,y)
                exp=np.insert(exp,exp.shape[0],np.nan)
        
        ar_ref[:,x]=ref
        ar_exp[:,x]=exp
        
    return ar_ref,ar_exp
    
def mixing(u_ar,d_ar):
    ar=np.full([u_ar.shape[0]*2,u_ar.shape[1]],np.nan)
    ar[::2]=u_ar
    ar[1::2]=d_ar
    ar2=np.delete(ar,0,0)
    ar3=np.delete(ar,ar.shape[0]-1 , 0)
    ar=(ar2+ar3)/2
    
    return ar


def complementer(ref_ar,diff_ar):
    max_ar=int(np.nanmax(ref_ar))
    diff_2=np.vstack([np.full(max_ar,-1),range(max_ar),np.zeros(max_ar)]).T
    
    
    for x in range(ref_ar.shape[1]):
        ar_loop=np.vstack([np.full_like(ref_ar[:,x],x),ref_ar[:,x],diff_ar[:,x]]).T
        ar_loop=ar_loop[~np.isnan(ar_loop).any(axis=1)]
        diff_2=np.concatenate([diff_2,ar_loop])
    
    diff_2[:,1]=diff_2[:,1].astype(int)
    diff_df=pd.DataFrame(diff_2)
    diff_df=diff_df.pivot_table(columns=0,index=1,values=2)
    diff_df=diff_df.interpolate(limit=50)
    
    diff_comp=diff_df.values
    
    return diff_comp