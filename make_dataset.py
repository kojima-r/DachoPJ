import scipy
import glob
import numpy as np
import json
import os
from multiprocessing import Pool

def process(args):
    filename=args["name"]
    feature=args["feature"]
    max_length=args["max_length"]
    print("[LOAD]",filename)
    f=np.load(filename)
    n=f.shape[1]
    if n>max_length:
        itr=0
        f_list=[]
        while itr<n:
            x=f[:,itr:itr+max_length]
            f_list.append(x)
            itr += max_length
        return filename,f_list
    else:
        return filename,[f]

def main():
    data=[]
    feature="spec"
    test_ratio=0.0
    max_length=10000
    filename_list=glob.glob("data_npy/*.npy")
    #feature="mfcc"
    feature="spec"
    for filename in filename_list:
        #name,_=os.path.splitext(os.path.basename(filename))
        data.append({"name":filename,"feature":feature,"max_length":max_length})

    p = Pool(64)
    results=p.map(process, data)
    p.close()

    result_list=[]
    for res in results:
        name,f_list=res
        for f in f_list:
            result_list.append((name,f))
            
    #ml=max(results)

    np.random.seed(1234)
    
    print("#sample:",len(result_list))
    ml=max([r[1].shape[1]  for r in result_list])
    print("max length:",ml)
    results=result_list
    feature_num=results[0][1].shape[0]
    n=len(results)
    print("results:",n)

    data=np.zeros((n,feature_num,ml),dtype=np.float32)
    step_data=np.zeros((n,),dtype=np.int32)
    #label_data=np.zeros((n,ml),dtype=np.int32)
    name_list=[]
    #label_mapping={}
    for i,r in enumerate(results):
        name,f=r
        s=f.shape[1]
        data[i,:,:s]=f
        step_data[i]=s
        """
        if label not in label_mapping:
            label_mapping[label]=len(label_mapping)
            label_data[i,:]=label_mapping[label]
        """
        name_list.append(name)
    print(data.shape)
    data=np.transpose(data,[0,2,1])
    ##
    all_idx=list(range(n))
    np.random.shuffle(all_idx)
    m=int(n*test_ratio)
    train_idx=all_idx[:n-m]
    test_idx=all_idx[n-m:]
    info={}
    info["pid_list_train"]=[name_list[i] for i in train_idx]
    info["pid_list_test"]=[name_list[i] for i in test_idx]
    ##
    os.makedirs("dataset",exist_ok=True)
    train_data=data[train_idx,:,:]
    train_step_data=step_data[train_idx]
    #train_label_data=label_data[train_idx,:]
    filename="dataset/train_data."+feature+".npy"
    np.save(filename,train_data)
    filename="dataset/train_step."+feature+".npy"
    np.save(filename,train_step_data)
    #filename="dataset/train_label."+feature+".npy"
    #np.save(filename,train_label_data)
    ##
    test_data=data[test_idx,:,:]
    test_step_data=step_data[test_idx]
    #test_label_data=label_data[test_idx,:]
    filename="dataset/test_data."+feature+".npy"
    np.save(filename,test_data)
    filename="dataset/test_step."+feature+".npy"
    np.save(filename,test_step_data)
    #filename="dataset/test_label."+feature+".npy"
    #np.save(filename,test_label_data)

    #print(ml)
    fp = open("dataset/info."+feature+".json", "w")
    json.dump(info, fp)
    #fp = open("dataset/label."+feature+".json", "w")
    #json.dump({v:k for k,v in label_mapping.items()}, fp)

if __name__ == '__main__':
    main()
