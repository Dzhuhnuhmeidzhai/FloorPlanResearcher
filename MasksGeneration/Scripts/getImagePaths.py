import os


#get paths to images
def getImagePaths(input_dir, target_dir_w, target_dir_dw, target_dir_r):
    #input image paths
    input_img_paths = sorted([
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
    ])  
    #target walls image paths
    target_imgw_paths = [
            os.path.join(target_dir_w, ((fname.split('/')[-1]).split('.')[0]+"_wall.png"))
            for fname in input_img_paths
    ]
    #target doors-windows image paths
    target_imgdw_paths = [
            os.path.join(target_dir_dw, ((fname.split('/')[-1]).split('.')[0]+"_close.png"))
            for fname in input_img_paths
    ]
    #target rooms image paths
    target_imgr_paths = [
            os.path.join(target_dir_r, ((fname.split('/')[-1]).split('.')[0]+"_rooms.png"))
            for fname in input_img_paths
    ] 
    return input_img_paths, target_imgw_paths, target_imgdw_paths, target_imgr_paths