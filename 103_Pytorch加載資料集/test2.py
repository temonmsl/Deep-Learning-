import os 

rootdir = '103/dataset/hymenoptera_data/hymenoptera_data/train'
subdir ='ants'
imgpath = os.listdir(os.path.join(rootdir,subdir))
outdir = 'ant_img'
lable = outdir.split('_')[0]#將 'ant_img' 分割並取第一部分，得到 'ant'。

for i in imgpath:
    revisename = i.split('.jpg')[0]
    with open(os.path.join(rootdir,outdir,"{}.txt".format(revisename)),'w') as f:
        f.write(lable)
