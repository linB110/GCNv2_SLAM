def split_line(f_rgbd):
    rgbd_split = f_rgbd.read().split()
    index_to_delete = []

    for i in range(len(rgbd_split)):
        if i % 2:
            index_to_delete.append(i)
    count = 0

    for index in index_to_delete:
        index = index - count
        rgbd_split.pop(index)
        count += 1

    return rgbd_split
 
def near_time_stamp(rgb_time_stamp, depth_time_stamp):
    near_rgb_stamp = []

    for index in depth_time_stamp:
        min_value = 1000000000
        min_jndex = ''
        depth_float = float(index)
        for jndex in rgb_time_stamp:
            rgb_float = float(jndex)
            if abs(depth_float - rgb_float) < min_value:
                min_value = abs(depth_float - rgb_float)
                min_jndex = jndex
        near_rgb_stamp.append(min_jndex)

    return near_rgb_stamp
 
 
if __name__ == '__main__':
 
    f_rgb = open("rgb.txt","r")
    f_depth = open("depth.txt","r")
    f_association = open('association.txt','w',encoding='utf-8')
    data_depth = f_depth.readlines()
    f_depth.seek(0,0)
 
    rgb_time_stamp = split_line(f_rgb)
    depth_time_stamp = split_line(f_depth)
 
    rgb_time_stamp = near_time_stamp(rgb_time_stamp, depth_time_stamp)
    
    for i in range(len(rgb_time_stamp)):
        str_association = rgb_time_stamp[i] + " " \
                            + "rgb/" + rgb_time_stamp[i] + ".png" + " " \
                            + data_depth[i]
        f_association.write(str_association)
 
    f_rgb.close()
    f_depth.close()
    f_association.close()
