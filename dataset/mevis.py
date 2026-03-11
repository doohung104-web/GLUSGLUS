import os
import json

def load_mevis_json(base_image_dir, is_train=True, set_name=None):
    # 默认划分
    data_split = "train" if is_train else "valid_u"
    if set_name is not None:
        data_split = set_name

    # ⭐ 尝试 DATASET/mevis/train/meta_expressions.json
    mevis_path = os.path.join(base_image_dir, "mevis", data_split, "meta_expressions.json")
    if os.path.isfile(mevis_path):
        root_dir = os.path.join(base_image_dir, "mevis")
    else:
        # 回退到 DATASET/train/meta_expressions.json
        root_dir = base_image_dir

    image_root = os.path.join(root_dir, data_split, 'JPEGImages')
    json_file = os.path.join(root_dir, data_split, 'meta_expressions.json')

    # 读取 JSON
    with open(str(json_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']

    videos = list(subset_expressions_by_video.keys())
    metas = []
    vid_list = []
    mask_dict = None

    # 训练集：需要 mask_dict
    if data_split == "train":
        mask_json = os.path.join(root_dir, data_split, 'mask_dict.json')
        with open(mask_json, 'r') as fp:
            mask_dict = json.load(fp)

        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            if len(vid_frames) < 4:
                continue
            vid_list.append(vid)

            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {
                    'str_id': f'mevis_{vid}_{exp_id}',
                    'video': vid,
                    'exp': exp_dict['exp'],
                    'obj_id': [int(x) for x in exp_dict['obj_id']],
                    'anno_id': exp_dict['anno_id'],
                    'frames': vid_frames,
                    'exp_id': exp_id,
                    'category': None,
                    'length': len(vid_frames),
                    'file_names': [
                        os.path.join(image_root, vid, frame + '.jpg')
                        for frame in vid_frames
                    ]
                }
                metas.append(meta)

    # 验证集 / valid_u
    else:
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            if len(vid_frames) < 4:
                continue
            vid_list.append(vid)

            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {
                    'str_id': f'mevis_{vid}_{exp_id}',
                    'video': vid,
                    'exp': exp_dict['exp'],
                    'obj_id': None,
                    'anno_id': None,
                    'frames': vid_frames,
                    'exp_id': exp_id,
                    'category': None,
                    'length': len(vid_frames),
                    'file_names': [
                        os.path.join(image_root, vid, frame + '.jpg')
                        for frame in vid_frames
                    ]
                }
                metas.append(meta)

    return vid_list, metas, mask_dict

    root_dir = os.path.dirname(os.path.dirname(json_file))   # 去掉 data_split 和文件名
    image_root=os.path.join(base_image_dir, data_split, 'JPEGImages')
    json_file=os.path.join(base_image_dir, data_split, 'meta_expressions.json')
    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0


    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    metas = []
    vid_list = []
    mask_dict = None
    if data_split == "train":
        mask_json = os.path.join(base_image_dir, data_split, 'mask_dict.json')
        with open(mask_json, 'r') as fp:
            mask_dict = json.load(fp)
            
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 4:
                continue
            vid_list.append(vid)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['str_id'] = f'mevis_{vid}_{exp_id}'
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = exp_dict['anno_id']
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = None
                meta['length'] = vid_len
                meta['file_names'] = [os.path.join(image_root, vid, vid_frames[i]+ '.jpg') for i in range(vid_len)]
                metas.append(meta)
    else:
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 4:
                continue
            vid_list.append(vid)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['str_id'] = f'mevis_{vid}_{exp_id}'
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = None
                meta['anno_id'] = None
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = None
                meta['length'] = vid_len
                meta['file_names'] = [os.path.join(image_root,  vid, vid_frames[i]+ '.jpg') for i in range(vid_len)]
                metas.append(meta)
    
    return vid_list, metas, mask_dict




